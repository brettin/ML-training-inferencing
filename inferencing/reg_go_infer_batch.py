import pickle
import numpy as np
import pandas as pd
import csv
import argparse

from keras.models import load_model
from keras import backend as K
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# a custom metric was used during training
def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def tf_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def auroc(y_true, y_pred):
    score = tf.py_func(
        lambda y_true, y_pred: roc_auc_score(y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
        [y_true, y_pred],
        'float32',
        stateful=False,
        name='sklearnAUC')
    return score


def load_pkl_list(pkl_list_path):
    with open(pkl_list_path) as f:
        pkl_list = []
        reader = csv.reader(f, delimiter=" ")
        for line in reader:
            # pkl_list.append([x.strip() for x in line])
            pkl_list.append(line[0])

    return pkl_list


def load_headers(desc_headers, train_headers):
    # get descriptor and training headers
    # the model was trained on 1613 features
    # the new descriptor files have 1826 features

    with open(desc_headers) as f:
        reader = csv.reader(f, delimiter=",")
        dh_row = next(reader)
        dh_row = [x.strip() for x in dh_row]

    dh_dict = {}
    for i in range(len(dh_row)):
        dh_dict[dh_row[i]] = i

    with open(train_headers) as f:
        reader = csv.reader(f, delimiter=",")
        th_list = next(reader)
        th_list = [x.strip() for x in th_list]

    return dh_dict, th_list


def load_dataset_from_pkl(dh_dict, th_list, pickle_file):
    # read the pickle descriptor file
    with open(pickle_file, 'rb') as pf:
        data = pickle.load(pf)
        df = pd.DataFrame(data).transpose()
        df.dropna(how='any', inplace=True)

    # build np array from pkl file
    cols = len(df.iloc[0][1])
    rows = df.shape[0]
    samples = np.empty([rows, cols], dtype='float32')

    for i in range(rows):
        a = df.iloc[i, 1]
        samples[i] = a

    samples = np.nan_to_num(samples)

    # build np array with reduced feature set
    reduced = np.empty([rows, len(th_list)], dtype='float32')
    i = 0
    for h in th_list:
        reduced[:, i] = samples[:, dh_dict[h]]
        i = i + 1

    # scale
    scaler = StandardScaler()
    df_x = scaler.fit_transform(reduced)

    return rows, df, df_x


def load_save_model(model_path):
    dependencies = {'r2': r2, 'tf_auc': tf_auc, 'auroc': auroc}
    model = load_model(model_path, custom_objects=dependencies)
    model.summary()
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.0001, momentum=0.9), metrics=['mae', r2])

    return model


def run_infer(model, rows, df, df_x, output_path):
    predictions = model.predict(df_x, batch_size=10000)
    assert(len(predictions) == rows)

    with open(output_path, "w") as f:
        for n in range(rows):
            if len(df.iloc[1, 0]) == 0:
                # IDENTIFIER_LIST is empty, use smile
                print("{},{},{}".format(df.index[n], predictions[n][0], df.index[n]), file=f)
            else:
                print("{},{},{}".format(df.iloc[n, 0][0], predictions[n][0], df.index[n]), file=f)


def main():
    psr = argparse.ArgumentParser(description='inferencing on descriptors')
    psr.add_argument('--in',  default='G17.input_test')
    psr.add_argument('--model',  default='model.h5')
    psr.add_argument('--dh',  default='../descriptor_headers.csv')
    psr.add_argument('--th',  default='../training_headers.csv')
    psr.add_argument('--out', default='', help='output directory')
    args = vars(psr.parse_args())
    print(args)

    dh_dict, th_list = load_headers(args['dh'], args['th'])
    model = load_save_model(args['model'])

    pkl_list = load_pkl_list(args['in'])
    for pkl in pkl_list:
        try:
            print(f'processing {pkl}')
            rows, df, df_x = load_dataset_from_pkl(dh_dict, th_list, pkl)
            run_infer(model, rows, df, df_x, f'{args["out"]}/{pkl.split("/")[-1]}')
        except UnicodeDecodeError:
            print(f'***** cannot process {pkl}')
            # pass


if __name__ == '__main__':
    main()
