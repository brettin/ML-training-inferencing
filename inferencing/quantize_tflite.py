import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3

import argparse
parser = argparse.ArgumentParser(description="create quantized models")
parser.add_argument("-d", default="./Models")
parser.add_argument("-o", default="./tflite_model")
args = parser.parse_args()
saved_model_dir = args.d
out_model = args.o

# Converting a SavedModel to a TensorFlow Lite model with no quantization
# It's now a TensorFlow Lite model, but it's still using 32-bit float values for all parameter data.

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
with open('{}.tflite'.format(out_model), 'wb') as f:
    f.write(tflite_model)


# Quantizing the tflite_model
# Using dynamic range quantization

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quant = converter.convert()
with open('{}_quant.tflite'.format(out_model), 'wb') as f:
    f.write(tflite_model_quant)


# Convert using float fallback quantization
d = np.load('X_train.npy')
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(d):
         yield[input_value]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
tflite_model_quant_float_fallback = converter.convert()

with open('{}_quant_float_fallback.tflite'.format(out_model), 'wb') as f:
    f.write(tflite_model_quant_float_fallback)



