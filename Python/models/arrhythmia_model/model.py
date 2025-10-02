#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    21-Sep-2025 06:32:02

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    sequenceinput = keras.Input(shape=(None,1))
    conv1d_1 = layers.Conv1D(128, 9, padding="same", name="conv1d_1_")(sequenceinput)
    batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1_")(conv1d_1)
    relu_1 = layers.ReLU()(batchnorm_1)
    dropout_1 = layers.Dropout(0.400000)(relu_1)
    conv1d_2 = layers.Conv1D(128, 7, padding="same", dilation_rate=2, name="conv1d_2_")(dropout_1)
    batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2_")(conv1d_2)
    relu_2 = layers.ReLU()(batchnorm_2)
    dropout_2 = layers.Dropout(0.400000)(relu_2)
    conv1d_3 = layers.Conv1D(128, 5, padding="same", dilation_rate=2, name="conv1d_3_")(dropout_2)
    batchnorm_3 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_3_")(conv1d_3)
    relu_3 = layers.ReLU()(batchnorm_3)
    dropout_3 = layers.Dropout(0.400000)(relu_3)
    globalavgpool1d = layers.GlobalAveragePooling1D(keepdims=False)(dropout_3)
    fc_1 = layers.Dense(32, name="fc_1_")(globalavgpool1d)
    relu_4 = layers.ReLU()(fc_1)
    dropout_4 = layers.Dropout(0.400000)(relu_4)
    fc_2 = layers.Dense(22, name="fc_2_")(dropout_4)
    softmax = layers.Softmax()(fc_2)

    model = keras.Model(inputs=[sequenceinput], outputs=[softmax])
    return model
