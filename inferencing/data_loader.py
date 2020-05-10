import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
import functools
import operator


class DataLoader(keras.utils.Sequence):
    def __init__(self, file_format=None, input_list=[]):
        self.input_list = input_list
        self.file_format = file_format
        self.labels = []

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        return self.load_dataset(self.input_list[idx])

    def load_dataset(self, file_name):
        if self.file_format == '.parquet':
            df = pd.read_parquet(file_name)
        elif self.file_format == '.feather':
            df = pd.read_feather(file_name)

        self.labels.append(df[['ID','SMILE']].values.tolist())

        return df.iloc[:, 2:1615]

    def get_labels(self):
        return functools.reduce(operator.iconcat, self.labels, [])

