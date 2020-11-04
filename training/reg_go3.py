import pandas as pd
import numpy as np
import os
import sys
import csv
import argparse

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, model_from_json, model_from_yaml
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path)

psr = argparse.ArgumentParser(description='input csv file')
psr.add_argument('--in',  default='in_file')
psr.add_argument('--ep',  type=int, default=400)
psr.add_argument('--saved_model_dir', default='SavedModelDir')
args=vars(psr.parse_args())
print(args)

data_path = args['in']
saved_model_dir = args['saved_model_dir']
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)


EPOCH = args['ep']
BATCH = 32
DR    = 0.1
#nb_classes = 2


def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def load_headers(desc_headers, train_headers):
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


def load_data():
    data_path = args['in']
    dh_dict, th_list = load_headers('./descriptor_headers.csv', './training_headers.csv')
    offset = 6  # descriptor starts at index 6
    desc_col_idx = [dh_dict[key] + offset for key in th_list]

    df = pd.read_parquet(data_path)
    df_y = df['reg'].astype('float32')
    df_x = df.iloc[:, desc_col_idx].astype(np.float32)

    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x)
    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size= 0.20, random_state=42)
    
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_data()
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('Y_train.npy', Y_train)
np.save('Y_test.npy', Y_test)

PS=X_train.shape[1]

inputs = Input(shape=(PS,), name="Input_Layer_1")
x = Dense(250, activation='elu', name="Dense_Layer_1")(inputs)
x = Dropout(DR, name="Dropout_Layer_1")(x)
x = Dense(125, activation='elu', name="Dense_Layer_2")(x)
x = Dropout(DR, name="Dropout_Layer_2")(x)
x = Dense(60, activation='elu', name="Dense_Layer_3")(x)
x = Dropout(DR, name="Dropout_Layer_3")(x)
x = Dense(30, activation='elu', name="Dense_Layer_4")(x)
x = Dropout(DR, name="Dropout_Layer_4")(x)
outputs = Dense(1, activation='relu', name="Output_Layer_1")(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss='mean_squared_error',
              optimizer=SGD(lr=0.0001, momentum=0.9),
              metrics=['mae',r2])


checkpointer = ModelCheckpoint(filepath='reg_go.autosave.model.h5',
        verbose=1, save_weights_only=False, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=20,
        verbose=1, mode='auto', epsilon=0.0001, cooldown=3, min_lr=0.000000001)
early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
csv_logger = CSVLogger('reg_go.training.log')


history = model.fit(X_train, Y_train,                             
                    batch_size=BATCH,
                    epochs=EPOCH,
                    verbose=1,
                    validation_data=(X_test, Y_test),
                    callbacks = [checkpointer, csv_logger, reduce_lr, early_stop])

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
print(history.history.keys())


model.save(saved_model_dir)

model_json = model.to_json()
with open("reg_go.model.json", "w") as json_file:
        json_file.write(model_json)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("reg_go.model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)


# serialize weights to HDF5
model.save_weights("reg_go.model.h5")
print("Saved model to disk")


# load json and create new model
json_file = open('reg_go.model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_json = model_from_json(loaded_model_json)


# load yaml and create new model
yaml_file = open('reg_go.model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model_yaml = model_from_yaml(loaded_model_yaml)


# load weights into new model and evaluate on test data
loaded_model_json.load_weights("reg_go.model.h5")
loaded_model_json.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_absolute_error'])
score_json = loaded_model_json.evaluate(X_test, Y_test, verbose=0)

print("Loaded json model from disk")
print('json Validation loss:', score_json[0])
print('json Validation mae:', score_json[1])

# load weights into new model and evaluate on test data
loaded_model_yaml.load_weights("reg_go.model.h5")
loaded_model_yaml.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_absolute_error'])
score_yaml = loaded_model_yaml.evaluate(X_test, Y_test, verbose=0)

print("Loaded yaml model from disk")
print('yaml Validation loss:', score_yaml[0])
print('yaml Validation mae:', score_yaml[1])

# predict using loaded yaml model on test and training data
predict_yaml_train = loaded_model_yaml.predict(X_train)
predict_yaml_test = loaded_model_yaml.predict(X_test)
pred_train = predict_yaml_train[:,0]
pred_test = predict_yaml_test[:,0]
np.savetxt("pred_train.csv", pred_train, delimiter=".", newline='\n', fmt="%.3f")
np.savetxt("pred_test.csv", pred_test, delimiter=",", newline='\n',fmt="%.3f")
np.savetxt("Y_train.csv", Y_train, delimiter=",", newline='\n',fmt="%.3f")
np.savetxt("Y_test.csv", Y_test, delimiter=",", newline='\n',fmt="%.3f")

print('Correlation prediction on test and Y_test:', np.corrcoef( pred_test, Y_test))
print('Correlation prediction on train and Y_train:', np.corrcoef( pred_train, Y_train))

