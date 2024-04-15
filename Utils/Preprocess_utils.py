import argparse
import os
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rank_bm25 import BM25L


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def preprocess_set_args():
    parser = argparse.ArgumentParser(description="Set Pre-process Arguments.")

    parser.add_argument('--seed', default='42', type=int, help='Set seed for initialization.')
    parser.add_argument('--pretrained_model', type=str, default='microsoft/graphcodebert-base')
    parser.add_argument('--code_path', type=str, default='Dataset/train_code')
    parser.add_argument('--truncation_side', type=str, default='left')

    args = parser.parse_args()
    return args

def preprocess_mkdir(args):
    model_name = args.pretrained_model
    model_name = model_name.replace('/', '_')

    if not os.path.exists(f"./Dataset/pp_train_{model_name}"):
        os.makedirs(f"./Dataset/pp_train_{model_name}")
    if not os.path.exists(f"./Dataset/pp_test_{model_name}"):
        os.makedirs(f"./Dataset/pp_test_{model_name}")

def remove_hidden_directory(directories):
    ''' Removes any files/folders starting with "." (e.g., `.DS_Store`, `.ipynb_checkpoints`). '''
    clean_directories = [directory for directory in directories if not directory.startswith('.')]

    return clean_directories

def clean_data(data, data_type):
    with open(data, 'r', encoding='utf-8') as file:
        cleaned_lines = []
        lines = file.readlines()
        multiLineCommentFlag = False

        for line in lines:
            if line.startswith('#include'):
                continue
            line = line.strip().replace('\t', '').split('//')[0].strip()
            line = re.sub(' +', ' ', line)

            if line == '':
                continue

            if line.startswith('/*'):
                multiLineCommentFlag = True

            if not multiLineCommentFlag:
                cleaned_lines.append(line)

            if line.endswith('*/'):
                multiLineCommentFlag = False
    
    concatenated_lines = ' '.join(cleaned_lines)
    return concatenated_lines

def make_pairs(train_df, tokenizer):
    codes = train_df['code'].tolist()
    problems = train_df['problem_num'].unique().tolist()
    problems.sort()

    tokenized_corpus = [tokenizer.tokenize(code, max_length=512, truncation=True) for code in codes]
    bm_25L = BM25L(tokenized_corpus)

def make_df(args):
    code_folders_path = args.code_path
    code_folders = remove_hidden_directory(os.listdir(code_folders_path))
    problem_nums, preprocessed_scripts = [], []

    for code_folder in tqdm(code_folders):
        scripts = remove_hidden_directory(os.listdir(os.path.join(code_folders_path, code_folder)))
        problem_num = scripts[0].split('_')[0]
        
        for script in scripts:
            script_file = os.path.join(code_folders_path, code_folder, script)
            cleaned_data = clean_data(script_file, data_type="dir")
            preprocessed_scripts.append(cleaned_data)
            
        problem_nums.extend([problem_num] * len(scripts))

    train_df = pd.DataFrame(data={'code': preprocessed_scripts, 'problem_num': problem_nums})

    return train_df        
        



            