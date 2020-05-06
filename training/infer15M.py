import pickle
import numpy as np
import pandas as pd
import csv
import argparse

from keras.models import Sequential, Model, load_model
from keras import backend as K
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


# hard code args for testing
# args={}
# args['in'] = '/projects/CVD_Research/datasets/15M/xcg.smi.desc.fix'
# args['out'] = 'out_file'
# args['model'] = '/projects/CVD_Research/brettin/March_30/DIR.ml.ADRP-ADPR_pocket1_dock.csv.reg.csv/reg_go.autosave.model.h5

psr = argparse.ArgumentParser(description='inferencing on descriptors')
psr.add_argument('--in',  default='in_file.csv')
psr.add_argument('--model',  default='model.h5')
psr.add_argument('--out', default='out_file.csv')
args=vars(psr.parse_args())

print(args)

# a custom metric was used during training
def r2(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred))
	SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
	return (1 - SS_res/(SS_tot + K.epsilon()))

def tf_auc(y_true, y_pred):
	auc = tf.metrics.auc(y_true, y_pred)[1]
	K.get_session().run(tf.local_variables_initializer())
	return auc

def auroc( y_true, y_pred ) :
	score = tf.py_func( lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
                        [y_true, y_pred],
                        'float32',
                        stateful=False,
                        name='sklearnAUC' )
	return score

dependencies={'r2' : r2, 'tf_auc' : tf_auc, 'auroc' : auroc }

''' The goal of fix_df is to modify the df to conform to a
single data frame format for use in inferencing. The desired
format is that the first column has a sample identifier, and
the rest of the columns have feature values.'''

def fix_df (df):
    # deal with header/no header, assumes header=None was passed to csv_read()
    if df.columns.dtype != np.int64:
        print ("did you read the csv file with header==None?")
        sys.exit("can not fix df, use header=None in call to read_csv")
    # assuming header=None, check if a header was actually in the csv file
    # we'll do this by checking the datatype of the 4th column
    elif df.iloc[:,4].dtype == object:
        print("looks like there was a header row in the file")
        df.drop(index=0, inplace=True)
        df.infer_objects()
    # deal with the first two columns being some combination of smile and name
    col_types={}
    for n in range(3):
        if df.iloc[:,n].dtype == object:
            print ("column {} looks like text".format(n))
            col_types[n]=object
        elif df.iloc[:,n].dtype == np.float64:
            print ("column {} looks like float64".format(n))
            col_types[n]=np.float64
        elif df.iloc[:,n].dtype == np.float32:
            print ("column {} looks like float32".format(n))
            col_types[n]=float32
    if col_types[0] == object and col_types[1] == object:
        if df.iloc[:,0].str.len().mean() < df.iloc[:,1].str.len().mean():
            print ("using col 0 as sample identifier")
            df.drop(columns=[1], inplace=True)
        else:
            print ("using col 1 as sample identifier")
            df.drop(columns=[0], inplace=True)
    return df


# read the csv descriptor file
# assumes there is a header
logging.info('reading {}'.format(args['in']))
df=pd.read_csv(args['in'], engine='python', header=None)
df=fix_df(df)

cols=df.shape[1] - 1
rows=df.shape[0]
samples=np.empty([rows,cols],dtype='float32')
samples=df.iloc[:,1:]
samples=np.nan_to_num(samples)

scaler = StandardScaler()
df_x = scaler.fit_transform(samples)
np.savetxt(args['out']+'mean_', scaler.mean_, delimiter=",")
np.savetxt(args['out']+'var_', scaler.var_, delimiter=",")

model = load_model(args['model'], custom_objects=dependencies)
model.summary()
model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.0001, momentum=0.9),metrics=['mae',r2])

predictions=model.predict(df_x)
assert(len(predictions) == rows)

with open (args['out'], "w") as f:
	for n in range(rows):
		print ( "{},{}".format(df.iloc[n,0],predictions[n][0]), file=f)

