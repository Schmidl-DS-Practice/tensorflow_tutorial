import os
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
keras = tf.keras

def main():

    # model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28,28)),
    #                                 keras.layers.Dense(128, activation='relu'),
    #                                 keras.layers.Dense(10)])

    # print(model.summary())

    # create model with functional API
    # Advantages:
    #   - Models with multiple inputs and outputs
    #   - Shared layers
    #   - Extract and reuse nodes in the graph of layers
    #   - Model are callable like layers (put model into sequential)
    # start by creating an Input node
    inputs = keras.Input(shape=(28,28))

    flatten = keras.layers.Flatten()
    dense1 = keras.layers.Dense(128, activation='relu')
    dense2 = keras.layers.Dense(10)

    x = flatten(inputs)
    x = dense1(x)
    outputs = dense2(x)

    # or with multiple outputs
    # dense2_2 = keras.layers.Dense(1)
    # outputs2 = dense2_2(x)
    # outputs = [output, outputs2]

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

    print(model.summary())

if __name__ == "__main__":
    main()