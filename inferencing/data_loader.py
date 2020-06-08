import pandas as pd
import keras
import numpy as np
import csv
import sys
import threading, queue


class DataLoader(keras.utils.Sequence):
    def __init__(self, file_format=None, input_list=[], label_file=None):
        self.input_list = input_list
        self.file_format = file_format
        self.first_called = False
        self.label_file = label_file
        self.labels_q = queue.Queue()
        def writer():
            with open(self.label_file, 'w') as f:
                label_writer = csv.writer(f, lineterminator='\n')
                for labels in iter(self.labels_q.get, None):
                    label_writer.writerows(labels)
        threading.Thread(target=writer).start()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        batch_x, batch_y = self.load_dataset(self.input_list[idx])

        if idx > 0 or (idx == 0 and self.first_called == False):
            labels = batch_y.values.tolist()
            self.labels_q.put(labels)

        if idx == 0 and self.first_called == True:
            self.labels_q.put(None)

        if idx == 0 and self.first_called == False:
            self.first_called = True

        return batch_x, batch_y

    def load_dataset(self, file_name):
        try:
            # print(f'loading {file_name}')
            if self.file_format == '.parquet':
                df = pd.read_parquet(file_name)
            elif self.file_format == '.feather':
                df = pd.read_feather(file_name, use_threads=False)

            batch_x = df.iloc[:, 2:1615]
            batch_y = df.iloc[:, 0:2]

            return batch_x, batch_y
        except:
            print("Unexpected error:", sys.exc_info()[0])
            print(f'error loading {file_name}')
            # return Empty matrix to continue
            batch_x = pd.DataFrame(data=np.zeros((10000, 1613)))
            batch_y = pd.DataFrame(data=np.zeros((10000, 2)))
            return batch_x, batch_y

    def close(self):
        self.labels_q.put(None)


class PredictLogger(keras.callbacks.Callback):
    def __init__(self, pred_file=None):
        self.pred_file = pred_file
        self.outputs_q = queue.Queue()
        def writer():
            with open(self.pred_file, 'w') as f:
                pred_writer = csv.writer(f, lineterminator='\n')
                for outputs in iter(self.outputs_q.get, None):
                    pred_writer.writerows(outputs[0])
        threading.Thread(target=writer).start()

    def on_predict_batch_end(self, batch, logs):
        pred = logs['outputs']
        self.outputs_q.put(pred)

    def on_predict_end(self, logs):
        self.outputs_q.put(None)

