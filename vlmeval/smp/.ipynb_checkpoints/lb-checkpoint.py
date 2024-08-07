import json
import pandas as pd
from collections import defaultdict
import gradio as gr
import copy as cp
import numpy as np
from .misc import listinstr

# CONSTANTS-URL
URL = "https://github.com/thecharm/BDoG.git"
VLMEVALKIT_README = 'https://raw.githubusercontent.com/thecharm/BDoG/main/README.md'
# CONSTANTS-CITATION
CITATION_BUTTON_TEXT = r"""@misc{zheng2024pictureworthgraphblueprint,
      title={A Picture Is Worth a Graph: Blueprint Debate on Graph for Multimodal Reasoning}, 
      author={Changmeng Zheng and Dayong Liang and Wengyu Zhang and Xiao-Yong Wei and Tat-Seng Chua and Qing Li},
      year={2024},
      eprint={2403.14972},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2403.14972}, 
}"""
CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
# CONSTANTS-TEXT
LEADERBORAD_INTRODUCTION = ""
# CONSTANTS-FIELDS
META_FIELDS = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model', 'OpenSource', 'Verified']
MAIN_FIELDS = ['MMBench_TEST_EN', 'MMBench_TEST_CN', 'CCBench', 'MME', 'SEEDBench_IMG', 'MMVet', 'MMMU_VAL', 'MathVista', 'HallusionBench', 'LLaVABench']
MMBENCH_FIELDS = ['MMBench_TEST_EN', 'MMBench_DEV_EN', 'MMBench_TEST_CN', 'MMBench_DEV_CN', 'CCBench']
MODEL_SIZE = ['<10B', '10B-20B', '20B-40B', '>40B', 'Unknown']
MODEL_TYPE = ['API', 'OpenSource', 'Proprietary']

LEADERBOARD_MD = {
}

from urllib.request import urlopen

def load_results():
    data = json.loads(urlopen(URL).read())
    return data

def nth_large(val, vals):
    return sum([1 for v in vals if v > val]) + 1

def format_timestamp(timestamp):
    return timestamp[:2] + '.' + timestamp[2:4] + '.' + timestamp[4:6] + ' ' + timestamp[6:8] + ':' + timestamp[8:10] + ':' + timestamp[10:12]

def model_size_flag(sz, FIELDS):
    if pd.isna(sz) and 'Unknown' in FIELDS:
        return True
    if pd.isna(sz):
        return False
    if '<10B' in FIELDS and sz < 10:
        return True
    if '10B-20B' in FIELDS and sz >= 10 and sz < 20:
        return True
    if '20B-40B' in FIELDS and sz >= 20 and sz < 40:
        return True
    if '>40B' in FIELDS and sz >= 40:
        return True
    return False

def model_type_flag(line, FIELDS):
    if 'OpenSource' in FIELDS and line['OpenSource'] == 'Yes':
        return True
    if 'API' in FIELDS and line['OpenSource'] == 'No' and line['Verified'] == 'Yes':
        return True
    if 'Proprietary' in FIELDS and line['OpenSource'] == 'No' and line['Verified'] == 'No':
        return True
    return False

def BUILD_L1_DF(results, fields):
    res = defaultdict(list)
    for i, m in enumerate(results):
        item = results[m]
        meta = item['META']
        for k in META_FIELDS:
            if k == 'Parameters (B)':
                param = meta['Parameters']
                res[k].append(float(param.replace('B', '')) if param != '' else None)
            elif k == 'Method':
                name, url = meta['Method']
                res[k].append(f'<a href="{url}">{name}</a>')
            else:
                res[k].append(meta[k])
        scores, ranks = [], []
        for d in fields:
            res[d].append(item[d]['Overall'])
            if d == 'MME':
                scores.append(item[d]['Overall'] / 28)
            else:
                scores.append(item[d]['Overall'])
            ranks.append(nth_large(item[d]['Overall'], [x[d]['Overall'] for x in results.values()]))
        res['Avg Score'].append(round(np.mean(scores), 1))
        res['Avg Rank'].append(round(np.mean(ranks), 2))

    df = pd.DataFrame(res)
    df = df.sort_values('Avg Rank')
    
    check_box = {}
    check_box['essential'] = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model']
    check_box['required'] = ['Avg Score', 'Avg Rank']
    check_box['all'] = check_box['required'] + ['OpenSource', 'Verified'] + fields
    type_map = defaultdict(lambda: 'number')
    type_map['Method'] = 'html'
    type_map['Language Model'] = type_map['Vision Model'] = type_map['OpenSource'] = type_map['Verified'] = 'str'
    check_box['type_map'] = type_map
    return df, check_box
        
def BUILD_L2_DF(results, dataset):
    res = defaultdict(list)
    fields = list(list(results.values())[0][dataset].keys())
    non_overall_fields = [x for x in fields if 'Overall' not in x]
    overall_fields = [x for x in fields if 'Overall' in x]
    if dataset == 'MME':
        non_overall_fields = [x for x in non_overall_fields if not listinstr(['Perception', 'Cognition'], x)]
        overall_fields = overall_fields + ['Perception', 'Cognition']
    
    for m in results:
        item = results[m]
        meta = item['META']
        for k in META_FIELDS:
            if k == 'Parameters (B)':
                param = meta['Parameters']
                res[k].append(float(param.replace('B', '')) if param != '' else None)
            elif k == 'Method':
                name, url = meta['Method']
                res[k].append(f'<a href="{url}">{name}</a>')
            else:
                res[k].append(meta[k])
        fields = [x for x in fields]
    
        for d in non_overall_fields:
            res[d].append(item[dataset][d])
        for d in overall_fields:
            res[d].append(item[dataset][d])

    df = pd.DataFrame(res)
    df = df.sort_values('Overall')
    df = df.iloc[::-1]
    
    check_box = {}
    check_box['essential'] = ['Method', 'Parameters (B)', 'Language Model', 'Vision Model']
    check_box['required'] = overall_fields
    check_box['all'] = non_overall_fields + overall_fields
    type_map = defaultdict(lambda: 'number')
    type_map['Method'] = 'html'
    type_map['Language Model'] = type_map['Vision Model'] = type_map['OpenSource'] = type_map['Verified'] = 'str'
    check_box['type_map'] = type_map
    return df, check_box