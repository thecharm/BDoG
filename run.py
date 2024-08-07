import torch
import torch.distributed as dist
from vlmeval.smp import *
from vlmeval.evaluate import multiple_choice_eval
from vlmeval.inference_multi import infer_data_job, prefetch_acc
from vlmeval.config import supported_VLM
from vlmeval.utils import dataset_URLs, abbr2full
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument("--model", type=str, nargs='+', default='instructblip_13b', required=False)
    parser.add_argument("--lmudata", type=str, default='', required=False)
    parser.add_argument("--openai", type=str, default='', required=False)
    parser.add_argument("--stage", type=str, default='BDebate', required=True)
    parser.add_argument("--nproc", type=int, default=4, help="Parallel API calling")
    parser.add_argument("--debate", type=int, default=2, required=True)
    parser.add_argument("--ignore", action='store_true', help="Ignore failed indices. ")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--prefetch", action='store_true')
    parser.add_argument("--kg_init", action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert len(args.data), "--data should be a list of data files"
    init_environ(args)
    
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=5400))

    for _, model_name in enumerate(args.model):
        model = None
        pred_root = model_name

        for i, dataset_name in enumerate(args.data):
            os.makedirs(f'results/{model_name}/{dataset_name}', exist_ok=True)
            
            logger_file = "results/{}/{}/{}_{}_{}_log.txt".format(args.model[0], args.data[0], args.model[0],args.data[0], args.stage)
            logger = get_logger(name='Multi', log_file=logger_file)
            logger.info(f"####- Begin -####\n{args.model[0]}: {args.data[0]}-{args.stage}")
            
            if dataset_name not in dataset_URLs:
                dataset_name = abbr2full(dataset_name)
            
            if dataset_name not in dataset_URLs:
                logger.error(f'Unknown dataset: {dataset_name}. ')
                continue

            result_file = f'results/{pred_root}/{dataset_name}/{model_name}_{dataset_name}_{args.stage}_DB{args.debate}.xlsx'
            
            if model is None:
                model = model_name # which is only a name

            model = infer_data_job(model, model_name=model_name, dataset_name=dataset_name, args=args,  logger=logger, ignore_failed=args.ignore)

            if rank == 0:
                time.sleep(3)
                res = None
                if listinstr(['MMBench'], dataset_name):
                    res = prefetch_acc(result_file)
                else:
                    logger.warning(f'{dataset_name} is not handled by prefetch score calculator')
                if res is not None:
                    logger.info(f'{model_name} prefetching: ')
                    logger.info(res)
                    dump(res, result_file.replace('.xlsx', '_prefetch.xlsx'))
        
                if listinstr(['MMBench','ScienceQA'], dataset_name):
                    multiple_choice_eval(result_file, dataset=dataset_name, model='chatgpt-0613', nproc=args.nproc, verbose=args.verbose)
                else:
                    logger.error(f'Dataset {dataset_name} is not handled by evaluator, will be skipped. ')

if __name__ == '__main__':
    main()
