#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    21-Sep-2025 06:31:21

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    sequenceinput = keras.Input(shape=(None,1))
    conv1d_1 = layers.Conv1D(64, 7, padding="same", name="conv1d_1_")(sequenceinput)
    batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1_")(conv1d_1)
    relu_1 = layers.ReLU()(batchnorm_1)
    dropout_1 = layers.Dropout(0.400000)(relu_1)
    conv1d_2 = layers.Conv1D(64, 5, padding="same", dilation_rate=2, name="conv1d_2_")(dropout_1)
    batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2_")(conv1d_2)
    relu_2 = layers.ReLU()(batchnorm_2)
    dropout_2 = layers.Dropout(0.400000)(relu_2)
    globalavgpool1d = layers.GlobalAveragePooling1D(keepdims=False)(dropout_2)
    fc_1 = layers.Dense(64, name="fc_1_")(globalavgpool1d)
    relu_3 = layers.ReLU()(fc_1)
    dropout_3 = layers.Dropout(0.400000)(relu_3)
    fc_2 = layers.Dense(2, name="fc_2_")(dropout_3)
    softmax = layers.Softmax()(fc_2)
    classoutput = softmax

    model = keras.Model(inputs=[sequenceinput], outputs=[classoutput])
    return model
