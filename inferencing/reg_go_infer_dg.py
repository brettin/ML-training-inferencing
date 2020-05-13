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
from data_loader import DataLoader, PredictLogger
from reg_go_infer_batch import load_pkl_list, load_save_model


def main():
    psr = argparse.ArgumentParser(description='inferencing on descriptors')
    psr.add_argument('--in',  default='G17.input_test')
    psr.add_argument('--model',  default='model.h5')
    psr.add_argument('--dh',  default='../descriptor_headers.csv')
    psr.add_argument('--th',  default='../training_headers.csv')
    psr.add_argument('--out', default='', help='output directory')
    args = vars(psr.parse_args())
    print(args)

    pkl_list = load_pkl_list(args['in'])
    pkl = pkl_list[0]
    ext = Path(pkl).suffix

    in_file = Path(args['in'])
    out_file = str(in_file.stem + in_file.suffix)
    label_file = Path(args['out'], out_file + '.label.csv')
    pred_file = Path(args['out'], out_file + '.pred.csv')

    if ext == '.parquet' or ext == '.feather':
        model = load_save_model(args['model'])
        data_gen = DataLoader(file_format=ext, input_list=pkl_list, label_file=label_file)

        predictions = model.predict_generator(
            data_gen,
            callbacks=[PredictLogger(pred_file=pred_file)],
        )
        data_gen.close()
    else:
        print(f"{ext} format is not supported")


if __name__ == '__main__':
    main()
