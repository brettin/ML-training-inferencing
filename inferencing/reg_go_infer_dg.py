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
from data_loader import DataLoader
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

    if ext == '.parquet' or ext == '.feather':
        model = load_save_model(args['model'])
        data_gen = DataLoader(file_format=ext, input_list=pkl_list)

        predictions = model.predict_generator(
            data_gen,
            workers=2,
        )
        labels = data_gen.get_labels()

        out_file = str(Path(args['in']).stem + Path(args['in']).suffix + '.csv')
        out_file = Path(args['out'], out_file)
        with open(out_file, "w") as f:
            for n in range(len(predictions)):
                print("{},{},{}".format(labels[n][0], predictions[n][0], labels[n][1]), file=f)
    else:
        print(f"{ext} format is not supported")


if __name__ == '__main__':
    main()
