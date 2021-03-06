import argparse
from pathlib import Path
import time
import functools
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from reg_go_infer_batch import load_headers, load_pkl_list as load_input_list
import ray

def main():
    psr = argparse.ArgumentParser(description='inferencing on descriptors')
    psr.add_argument('--in',  default=None, help='input file list')
    psr.add_argument('--dh',  default='../descriptor_headers.csv')
    psr.add_argument('--th',  default='../training_headers.csv')
    psr.add_argument('--out', default='/scratch/hsyoo/COVID', help='output directory')
    psr.add_argument('--wk', default=1, type=int)
    args = vars(psr.parse_args())
    # print(args)

    _start = time.time() 

    dh_dict, th_list = load_headers(args['dh'], args['th'])
    input_list = load_input_list(args['in'])
    ext = Path(input_list[0]).suffix

    if ext == '.csv':
        # compute column indexies
        offset = 3 #  descriptor starts at index 3
        desc_col_idx = [ dh_dict[key] + offset for key in th_list ]

        ray.init(num_cpus=args['wk'])
        futures = [process.remote(desc_col_idx, th_list, args, input_list[i]) for i in range(len(input_list))]
        results = ray.get(futures)
        total = functools.reduce(lambda a, b: a + b, results, 0)
    elif ext == '.gz':
        ray.init(num_cpus=args['wk'])
        futures = [process_csv_gz.remote(th_list, args, input_list[i]) for i in range(len(input_list))]
        results = ray.get(futures)
        total = functools.reduce(lambda a, b: a + b, results, 0)

    _end = time.time()
    print(f'Total elapsed: {_end - _start} and processed {total} files')


@ray.remote
def process(desc_col_idx, desc_col_names, args, input_file):
    scaler = StandardScaler()

    try:
        df = pd.read_csv(input_file, header=None, low_memory=False)
        df.fillna(0.0, inplace=True)
        col_id = df[0].map(str) + '.' + df[1].map(str)
        col_smile = df[2].map(str)
        col_desc = df[desc_col_idx].astype(np.float32)
    
        # scaling
        col_desc = scaler.fit_transform(col_desc)

        new_df = pd.DataFrame(data=col_desc, columns=desc_col_names)
        new_df.insert(0, column='SMILE', value=col_smile)
        new_df.insert(0, column='ID', value=col_id)
 
        file_name = Path(input_file).stem
        save_path = str(Path(args['out'], f'{file_name}.feather'))
        new_df.to_feather(save_path)
        # print(f'save {save_path}')

        return 1
    except:
        print(f'Error in processing {input_file}')
        print("Unexpected error:", sys.exc_info()[0])
        return 0


@ray.remote
def process_csv_gz(desc_col_names, args, input_file):
    scaler = StandardScaler()

    try:
        df = pd.read_csv(input_file, header=None, low_memory=False)
        df.fillna(0.0, inplace=True)
        col_id = df.iloc[:, 0].map(str) + '.' + df.iloc[:, 1].map(str)
        col_smile = df.iloc[:, 2].map(str)
        col_desc = df.iloc[:, 3:1616].astype(np.float32)

        # scaling
        col_desc = scaler.fit_transform(col_desc)

        new_df = pd.DataFrame(data=col_desc, columns=desc_col_names)
        new_df.insert(0, column='SMILE', value=col_smile)
        new_df.insert(0, column='ID', value=col_id)

        file_name = Path(input_file).stem
        file_name = Path(file_name).stem
        save_path = str(Path(args['out'], f'{file_name}.feather'))
        new_df.to_feather(save_path)
        # print(f'save {save_path}')

        return 1
    except:
        print(f'Error in processing {input_file}')
        print("Unexpected error:", sys.exc_info()[0])
        return 0
    

if __name__ == '__main__':
    main()

