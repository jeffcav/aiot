import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def mlp(solver='sgd', layers_sizes=[], activations=[]):
    x = keras.Input(shape=(77760,))

    y = x
    for layer_size, activation in zip(layers_sizes, activations):
        y = layers.Dense(layer_size, activation=activation)(y)

    model = keras.Model(x, y)
    model.compile(optimizer=solver, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
