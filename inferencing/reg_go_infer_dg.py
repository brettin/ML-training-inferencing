import pickle
import numpy as np
import pandas as pd
import csv
import argparse
from pathlib import Path

from keras.models import load_model
from keras import backend as K
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from data_loader import DataLoaderCancer, PredictLogger
from reg_go_infer_batch import load_pkl_list, load_save_model


def main():
    psr = argparse.ArgumentParser(description='inferencing on descriptors')
    psr.add_argument('--in',  default='G17.input_test')
    psr.add_argument('--model',  default='model.h5')
    psr.add_argument('--out', default='', help='output directory')
    psr.add_argument('--cl', default='cell_lines.csv', help='gene expression of cell lines')
    args = vars(psr.parse_args())
    print(args)

    pkl_list = load_pkl_list(args['in'])
    pkl = pkl_list[0]
    ext = Path(pkl).suffix

    in_file = Path(args['in'])
    out_file = str(in_file.stem + in_file.suffix)
    # label_file = Path(args['out'], out_file + '.label.csv')
    # pred_file = Path(args['out'], out_file + '.pred.csv')

    # load cell lines
    df_cells = pd.read_csv(args['cl'])

    if ext == '.parquet' or ext == '.feather':
        model = load_save_model(args['model'])

        for index, row in df_cells.iterrows():
            cl_label = row[0]
            cl_gene_exp = row[1:943]

            label_file = Path(args['out'], f'{out_file}.{cl_label}.label.csv')
            pred_file = Path(args['out'], f'{out_file}.{cl_label}.pred.csv')

            data_gen = DataLoaderCancer(file_format=ext, input_list=pkl_list, label_file=label_file, gene_exp_label=cl_label, gene_exp=cl_gene_exp)

            predictions = model.predict_generator(
                data_gen,
                max_queue_size=100,
                callbacks=[PredictLogger(pred_file=pred_file)],
            )

            data_gen.close()

            # post processing
            df_label = pd.read_csv(label_file, header=None, usecols=[0, 1], names=['CELL', 'DRUG'])
            df_pred = pd.read_csv(pred_file, header=None, names=['SCORE'])
            df_combined = pd.concat([df_label, df_pred], axis=1)
            combined_file = Path(args['out'], f'{out_file}.{cl_label}.csv.gz')
            df_combined.to_csv(combined_file, index=False, header=False, compression='gzip')
            label_file.unlink()
            pred_file.unlink()
    else:
        print(f"{ext} format is not supported")


if __name__ == '__main__':
    main()
