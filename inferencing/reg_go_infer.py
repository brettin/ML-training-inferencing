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

# hard code args for testing
#args={}
#args['evaluate'] = False
#args['in'] = '2019q3-4_Enamine_REAL_01.smi.chunk-0-10000.pkl'
#args['dh'] = 'descriptor_headers'
#args['th'] = 'training_headers'
#args['out'] = 'out_file'
#args['model' = 'agg_attn.autosave.model.h5'

psr = argparse.ArgumentParser(description='inferencing on descriptors')
psr.add_argument('--in',  default='in_file.pkl')
psr.add_argument('--model',  default='model.h5')
psr.add_argument('--dh',  default='descriptor_headers.csv')
psr.add_argument('--th',  default='training_headers.csv')
psr.add_argument('--out', default='out_file.csv')
args=vars(psr.parse_args())

print(args)

# get descriptor and training headers
# the model was trained on 1613 features
# the new descriptor files have 1826 features

with open (args['dh']) as f:
    reader = csv.reader(f, delimiter=",")
    drow = next(reader)
    drow = [x.strip() for x in drow]

tdict={}
for i in range(len(drow)):
    tdict[drow[i]]=i

f.close()
del reader

with open (args['th']) as f:
    reader = csv.reader(f, delimiter=",")
    trow = next(reader)
    trow = [x.strip() for x in trow]

f.close()
del reader


# read the pickle descriptor file

pf=open(args['in'], 'rb')
data=pickle.load(pf)
df=pd.DataFrame(data).transpose()
df.dropna(how='any', inplace=True)
pf.close()

# build np array from pkl file

cols=len(df.iloc[0][1])
rows=df.shape[0]
samples=np.empty([rows,cols],dtype='float32')

for i in range(rows):
        a=df.iloc[i,1]
        samples[i]=a

samples=np.nan_to_num(samples)

# build np array with reduced feature set

reduced=np.empty([rows,len(trow)],dtype='float32')
i=0
for h in trow:
    reduced[:,i]=samples[:,tdict[h] ]
    i=i+1

# del samples

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
scaler = StandardScaler()
df_x = scaler.fit_transform(reduced)

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
model = load_model(args['model'], custom_objects=dependencies)
model.summary()
model.compile(loss='mean_squared_error',optimizer=SGD(lr=0.0001, momentum=0.9),metrics=['mae',r2])

predictions=model.predict(df_x)
assert(len(predictions) == rows)

with open (args['out'], "w") as f:
    for n in range(rows):
        if len(df.iloc[1,0]) == 0:
            # IDENTIFIER_LIST is empty, use smile
            print ( "{},{},{}".format(df.index[n],predictions[n][0],df.index[n] ), file=f)
        else:
            print ( "{},{},{}".format(df.iloc[n,0][0],predictions[n][0],df.index[n] ), file=f)


