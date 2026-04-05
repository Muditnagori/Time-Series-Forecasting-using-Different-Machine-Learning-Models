import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, Dropout, GlobalAveragePooling1D

def transformer_block(inputs):
    x = MultiHeadAttention(num_heads=2, key_dim=32)(inputs, inputs)
    x = Dropout(0.1)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    return x

def build_transformer(seq_length):
    inputs = Input(shape=(seq_length, 1))
    x = transformer_block(inputs)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model