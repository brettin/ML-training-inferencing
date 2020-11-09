import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras.models import load_model

saved_model_dir = "3CLPro_7BQY_A_1_F/saved_model_dir"

# Quantize a model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quant = converter.convert()

# Save quantized model
out_model = '3CLPro_7BQY_A_1_F/model_quant'
with open('{}_quant.tflite'.format(out_model), 'wb') as f:
    f.write(tflite_model_quant)

# Get access to the tflite model weights
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
all_tensor_details = interpreter.get_tensor_details()
interpreter.allocate_tensors()

for tensor_item in all_tensor_details:
  print("Tensor Item Name %s:" % tensor_item["name"])
  print("Tensor Item Index %i:" % tensor_item["index"])
  print("Tensor Shape {}".format( np.asarray(interpreter.tensor(tensor_item["index"])()).shape ) )
  print(interpreter.tensor(tensor_item["index"])())


def get_min_max(data):
	#data = np.load(fname)
	return np.amin(data, axis=1),np.amax(data, axis=1)


# Load keras model

import numpy as np
from tensorflow import keras
saved_model_dir='ADRP_6W02_A_1_H/saved_model_dir'
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

custom_objects = {'r2' : r2}
keras_model = keras.models.load_model(saved_model_dir, custom_objects=custom_objects)

# Helper stuff to view weights inside a keras layer
np.asarray(keras_model.get_layer(name='Dense_Layer_1').get_weights())

### Assign quantized tensors to keras model

# all_tensor_details
# Index 6 -> Dense_Layer_1 (250, 1613)
# Index 7 -> Dense_Layer_2 (125, 250)
# Index 8 -> Dense_layer_3 (60, 125)
# Index 9 -> Dense_Layer_4 (30, 60)

qw1 = interpreter.tensor(6)().transpose()
qw2 = interpreter.tensor(7)().transpose()
qw3 = interpreter.tensor(8)().transpose()
qw4 = interpreter.tensor(9)().transpose()

# model.layers
# Index 1 -> Dense_Layer_1
# Index 3 -> Dense_Layer_2
# Index 5 -> Dense_Layer_3
# Index 7 -> Dense_Layer_4

mw1 = keras_model.layers[1].get_weights()
mw2 = keras_model.layers[3].get_weights()
mw3 = keras_model.layers[5].get_weights()
mw4 = keras_model.layers[7].get_weights()

mw1[0] = qw1
mw2[0] = qw2
mw3[0] = qw3
mw4[0] = qw4

keras_model.layers[1].set_weights(mw1)
keras_model.layers[3].set_weights(mw2)
keras_model.layers[5].set_weights(mw3)
keras_model.layers[7].set_weights(mw4)


