import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
keras = tf.keras
import numpy as np
import matplotlib.pyplot as plt


def main():
    mnist = keras.datasets.mnist
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    print(xtrain.shape, ytrain.shape)

    # normalize
    xtrain, xtest = xtrain/255, xtest/255

    # model
    model = keras.models.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                                  keras.layers.Dense(128, activation="relu"),
                                  keras.layers.Dense(10)])
    print(model.summary())


if __name__ == "__main__":
    main()