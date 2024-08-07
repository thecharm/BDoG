# -*- coding: utf-8 -*-
import torch 
import torch.distributed as dist
import random
import datetime
from vlmeval.config import supported_VLM
from vlmeval.utils import TSVDataset, track_progress_rich, split_MMMU, Debate_VLM
from vlmeval.smp import *
import logging
import numpy as np

FAIL_MSG = 'Failed to obtain answer via API.'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, nargs='+', required=True)
    parser.add_argument("--nproc", type=int, default=4, required=True)
    parser.add_argument("--debate", type=int, default=2, required=True)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

def check_identical_elements_2(lst):
    return len(set(lst)) == 1

def find_duplicates(lst):
    return list(set([x for x in lst if lst.count(x) > 1]))

def infer_data(model_name, dataset_name, out_file, logger, kg_init, stage="base", debate=2, verbose=False, api_nproc=4):

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        
    # Dataset init
    rank, world_size = get_rank_and_world_size()
    if rank == 0:
        dataset = TSVDataset(dataset_name)
    if world_size > 1:
        dist.barrier()
    dataset = TSVDataset(dataset_name)
    indices = list(range(rank, len(dataset), world_size))
    lt = len(indices)
    data = dataset.data.iloc[indices]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        return 
    data = data[~data['index'].isin(res)]
    lt = len(data)

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    
    for i in tqdm(range(lt)):
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        if stage != "base_line":
            struct = dataset.build_prompt_multi(data.iloc[i])
        else:
            struct = dataset.build_prompt(data.iloc[i])
            
        response = Debate_VLM(stage, model, struct, dataset_name, debate, kg_init, logger)
        torch.cuda.empty_cache()
        
        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)
    
    dump(res, out_file)
    return model

def prefetch_acc(result_file):
    data = load(result_file)
    from vlmeval.evaluate.multiple_choice import build_choices, can_infer
    tot = defaultdict(lambda: 0)
    match = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        tot['Overall'] += 1
        tot[cate] += 1
        choices = build_choices(item)
        matched = can_infer(item['prediction'], choices)
        if matched:
            match['Overall'] += 1
            match[cate] += 1
            if matched == item['answer']:
                hit['Overall'] += 1
                hit[cate] += 1
    res = defaultdict(list)
    for k in tot.keys():
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['match'].append(match[k])
        res['hit'].append(hit[k])
        res['match_rate'].append(match[k] / tot[k] * 100)
        if match[k] == 0:
            res['acc'].append(0)
        else:
            res['acc'].append(hit[k] / match[k] * 100)
    res = pd.DataFrame(res)
    return res

def infer_data_job(model, model_name, dataset_name, args, logger, ignore_failed=False):

    result_ = f'results/{model_name}/{dataset_name}/'
    result_file = result_ + f'{model_name}_{dataset_name}_{args.stage}_DB{args.debate}.xlsx'
    rank, world_size = get_rank_and_world_size()   
    tmpl = result_ + '{}' + f'{world_size}_{dataset_name}_{args.stage}_DB{args.debate}.pkl'
    out_file = tmpl.format(rank)

    if not osp.exists(result_file):
        model = infer_data(model, dataset_name=dataset_name, out_file=out_file, logger=logger, kg_init=args.kg_init, stage=args.stage, debate=args.debate, verbose=args.verbose)
        if world_size > 1:
            dist.barrier()

        if rank == 0:
            data_all = {}
            for i in range(world_size):
                data_all.update(load(tmpl.format(i)))

            data = TSVDataset(dataset_name).data
            print(len(data_all))
            print(len(data))
            assert len(data_all) == len(data)
            data['prediction'] = [str(data_all[x]) for x in data['index']]
            data.pop('image')
            
            dump(data, result_file)             
            for i in range(world_size):
                os.remove(tmpl.format(i))
        return model
    else:
        data = load(result_file)
        failed_set = []
        data['prediction'] = [str(x) for x in data['prediction']]
        for idx, pred in zip(data['index'], data['prediction']):
            if FAIL_MSG in str(pred):
                failed_set.append(idx)
        if len(failed_set) and (not ignore_failed):
            print(f'{len(failed_set)} records failed in the original result file {result_file}. ')
            assert rank == 0 and world_size == 1
            failed_set = set(failed_set)
            answer_map = {x: y for x, y in zip(data['index'], data['prediction'])}
            res = infer_data_api(model_name, dataset_name, failed_set, api_nproc=args.api_nproc)
            answer_map.update(res)
            data['prediction'] = [str(answer_map[x]) for x in data['index']]
            dump(data, result_file)
        return model_name

def main():
    logger = get_logger('Inference')

    args = parse_args()
    assert len(args.data), "--data should be a list of data files"

    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))

    for _, model_name in enumerate(args.model):
        model = None
        os.makedirs(model_name, exist_ok=True)
        pred_root = model_name

        for i, dataset_name in enumerate(args.data):

            result_file = f'{pred_root}/{dataset_name}/{model_name}_{dataset_name}.xlsx'
            if model is None:
                model = model_name # which is only a name
            model = infer_data_job(model, model_name=model_name, dataset_name=dataset_name, verbose=args.verbose, api_nproc=args.nproc)
                         
            if rank == 0 and listinstr(['MMBench','ScienceQA'], dataset_name):
                time.sleep(3)
                res = prefetch_acc(result_file)
                print(model_name, res)
                dump(res, result_file.replace('.xlsx', '_prefetch.xlsx'))

if __name__ == '__main__':
    main()
