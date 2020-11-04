import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
from datetime import datetime

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3

import argparse
parser = argparse.ArgumentParser(description="tflite model inferencing")
parser.add_argument("--model", default="./tflite_model.tflite")
args = parser.parse_args()
saved_model = args.model

# Load the inferencing samples into memory
X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')


# Load the TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter = tf.lite.Interpreter(model_path=saved_model)
all_tensor_details = interpreter.get_tensor_details()


# Get input and output tensors (information on input and output layers).
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Resize the input and output tensors to match the sample batch size
interpreter.resize_tensor_input(output_details[0]['index'],X_train.shape)
interpreter.resize_tensor_input(input_details[0]['index'],X_train.shape)


# Now allocate the tensors
interpreter.allocate_tensors()


# Show information on the tensors
for tensor_item in all_tensor_details:
  print("Tensor Item Name %s:" % tensor_item["name"])
  print("Tensor Item Index %i:" % tensor_item["index"])
  print("Tensor Shape {}".format( np.asarray(interpreter.tensor(tensor_item["index"])()).shape ) )
  print(interpreter.tensor(tensor_item["index"])())


# Start inferencing
print("started inferencing {}".format(datetime.now()))
interpreter.set_tensor(input_details[0]['index'], X_train)
interpreter.invoke()
print("finished inferencing {}".format(datetime.now()))


# For inferring on one sample in a set of samples, don't resize the
# input and output tensors above, and run this for loop over X_train
# for data in X_train:
#     interpreter.set_tensor(input_details[0]['index'], [data])
#     interpreter.invoke()


# Get predictions
output_data = interpreter.get_tensor(output_details[0]['index'])


if True:
  from sklearn.metrics import mean_absolute_error
  from sklearn.metrics import mean_squared_error
  print("MAE: {}".format(mean_absolute_error(Y_train,output_data[:,0])))
  print("MSE: {}".format(mean_squared_error(Y_train,output_data[:,0])))

