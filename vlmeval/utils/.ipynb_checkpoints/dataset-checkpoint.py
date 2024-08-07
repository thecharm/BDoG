import pandas as pd
import hashlib
from ..smp import *
from .dataset_config import dataset_URLs, dataset_md5_dict, img_root_map, DATASET_TYPE
from .custom_prompt import CustomPrompt
from .base_prompt import create_one_example
import csv

def isliststr(s):
    return (s[0] == '[') and (s[-1] == ']')

def check_md5(data_path, dataset):
    try:
        with open(data_path, 'rb') as f:
            hash = hashlib.new('md5')
            for chunk in iter(lambda: f.read(2**20), b''):
                hash.update(chunk)
        if str(hash.hexdigest()) == dataset_md5_dict[dataset]:
            return True
        else:
            warnings.warn('this data file is incomplete, so it needs to be downloaded again.')
            return False
    except:
        return False

    
def split_MMMU(struct):
    assert 'image' in struct and 'text' in struct
    text, images = struct['text'], struct['image']
    text_segs = text.split('<image ')
    segs = [text_segs[0]]
    for i, seg in enumerate(text_segs):
        if i == 0:
            continue
        assert istype(seg[0], int) and seg[1] == '>'
        image_idx = int(seg[0]) - 1
        segs.append(images[image_idx])
        segs.append(seg[2:])
    return segs

def init_prompt_multi(struct, format_):
    question = struct['text']['question'] if 'question' in struct['text'] else 'none'
    context = struct['text']['hint'] if 'hint' in struct['text'] else 'none'
    options = struct['text']['options'] if 'options' in struct['text'] else 'none'
    answer = struct['debate_ans'] if 'debate_ans' in struct else 'none'
    knowledge = struct['kg'] if 'kg' in struct else 'none'
    image_path = struct['image'] if 'image' in struct else 'none'
    
    prompt = create_one_example(format_, question, context.replace("\n"," "), options, answer, knowledge, image_path)
    return prompt

class TSVDataset(CustomPrompt):
    
    def __init__(self, dataset='MMBench_DEV_EN', img_root=None, skip_noimg=True):

        self.data_root = LMUDataRoot()
        assert osp.exists(self.data_root)

        self.dataset = dataset
        self.dataset_type = DATASET_TYPE(dataset)

        url = dataset_URLs[dataset]
        file_name = url.split('/')[-1]
        data_path = osp.join(self.data_root, file_name)
        print(data_path)

        if osp.exists(data_path) and md5(data_path) == dataset_md5_dict[dataset]:
            print("Dateset is Download: ",data_path)
            pass
        else:
            warnings.warn("The dataset tsv is not downloaded")
            download_file(url, data_path)

        data = load(data_path)
        if dataset=="ScienceQA_TEST":
            kg_file = "/code/BDoG/data/kg_init/scienceqa_test_kg_gpt4.json" 
            kg_base = json.load(open(kg_file))
            kg_base = list(kg_base.values())
        elif dataset=="MMBench_DEV_EN":
            kg_file = "/code/BDoG/data/kg_init/MMBench_DEV_EN_s.json" 
            kg_base = json.load(open(kg_file))
            kg_base = list(kg_base.values())
        else:
            kg_base = ['none' for i in range(len(data))]
            
        self.skip_noimg = skip_noimg
        if skip_noimg:
            data = data[~pd.isna(data['image'])]

        # Prompt for Captioning
        if listinstr(['COCO'], dataset):
            data['question'] = ['Please describe this image in general. Directly provide the description, do not include prefix like "This image depicts". '] * len(data)

        data['index'] = [str(x) for x in data['index']]
        data['image'] = [str(x) for x in data['image']]
        ## Add kg_init
        data['kg'] = kg_base
        
        image_map = {x: y for x, y in zip(data['index'], data['image'])}
        for k in image_map:
            if len(image_map[k]) <= 64:
                idx = image_map[k]
                assert idx in image_map and len(image_map[idx]) > 64
                image_map[k] = image_map[idx]
    
        data['image'] = [
            eval(image_map[k]) if isliststr(image_map[k]) else image_map[k] 
            for k in data['index']
        ]
        if 'image_path' in data:
            data['image_path'] = [
                eval(pths) if isliststr(pths) else pths for pths in data['image_path']
            ]
        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]
            
        self.data = data

        img_root = img_root if img_root is not None else osp.join('images', img_root_map[dataset])
        os.makedirs(img_root, exist_ok=True)
        self.img_root = img_root
        
#         self.set_file()
#         print("#### Save: /code/VLMEvalKit-main/data/CCBench_Fin.tsv")

    def __len__(self):
        return len(self.data)
    
    def set_file(self,data_in,data_out):
        dataset = self.dataset    
        print("##START Processing: ", dataset)
        with open(data_in, 'r') as in_file:
            with open(data_out, 'w', newline='') as out_file:
                reader = csv.DictReader(in_file, delimiter='\t')
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(out_file, fieldnames=fieldnames, delimiter='\t')

                writer.writeheader()
                for i, row in enumerate(tqdm(reader)):
                    # 给每一行增加一个键值对
                    line = self.data.iloc[i]
                    tgt_path = self.dump_image(line, dataset)
                    row['image'] = tgt_path
                    writer.writerow(row)
                print(writer.fieldnames)
        
    def build_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)

        prompt = line['question']
        if DATASET_TYPE(dataset) == 'multi-choice':
            question = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = 'Options:\n'
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = ''
            if hint is not None:
                prompt += f'Hint: {hint}\n'
            prompt += f'Question: {question}\n'
            if len(options):
                prompt += options_prompt
                prompt += "Answer with the option's letter from the given choices directly.\n"
        
        return dict(image=tgt_path, text=prompt)
    
    
    def build_prompt_multi(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)
        prompt = {}
        prompt['index'] = line['index']
        prompt['answer'] = line['gpt4_ans'] if dataset == "LLaVABench" else line['answer']
        prompt['question'] = line['question']
        if 'kg' in line:
            prompt['kg'] = str(line['kg'])[:350]
        else:
            prompt['kg'] = 'none'
        if DATASET_TYPE(dataset) == 'multi-choice':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            choise = []
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
                choise.append(item)
                
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if dataset == "LLaVABench":
                hint = line['caption'] if ('caption' in line and not pd.isna(line['caption'])) else None
            
            if hint is not None:
                prompt['hint'] = f'{hint}'
            if len(options):
                prompt['options'] = options_prompt
#                 prompt['options'] += 'Please select the correct answer from the options above.'
#                 prompt['options'] += "Answer with the option's letter from the given choices directly"
                prompt['choise'] = choise
#         print(tgt_path)
        return dict(image=tgt_path, text=prompt)
    
    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        mmqa_display(line)
    
