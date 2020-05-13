import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
import csv
import functools
import operator


class DataLoader(keras.utils.Sequence):
    def __init__(self, file_format=None, input_list=[], label_file=None):
        self.input_list = input_list
        self.file_format = file_format
        self.label_file = open(label_file, 'w')
        self.label_writer = csv.writer(self.label_file, lineterminator='\n')
        # self.labels = []

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        return self.load_dataset(self.input_list[idx])

    def load_dataset(self, file_name):
        try:
            # print(f'loading {file_name}')
            if self.file_format == '.parquet':
                df = pd.read_parquet(file_name)
            elif self.file_format == '.feather':
                df = pd.read_feather(file_name, use_threads=False)

            batch_x = df.iloc[:, 2:1615]
            batch_y = df.iloc[:, 0:2]

            # save label to file
            labels = batch_y.values.tolist()
            self.label_writer.writerows(labels)

            return batch_x, batch_y
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print(f'error loading {file_name}')

    # def get_labels(self):
    #    return functools.reduce(operator.iconcat, self.labels, [])

    def close(self):
        self.label_file.close()


class PredictLogger(keras.callbacks.Callback):
    def __init__(self, pred_file=None):
        self.pred_file = open(pred_file, 'w')
        self.pred_writer = csv.writer(self.pred_file, lineterminator='\n')

    def on_predict_batch_end(self, batch, logs):
        pred = logs['outputs'][0]
        self.pred_writer.writerows(pred)

    def on_predict_end(self, logs):
        self.pred_file.close()

