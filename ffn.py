import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
keras = tf.keras
layers = keras.layers
models = keras.models
import numpy as np
import matplotlib.pyplot as plt


def main():
    mnist = keras.datasets.mnist
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    print(xtrain.shape, ytrain.shape)

    # normalize
    xtrain, xtest = xtrain/255, xtest/255

    # model
    model = models.Sequential([layers.Flatten(input_shape=(28,28)),
                               layers.Dense(128, activation="relu"),
                               layers.Dense(10)])
    print(model.summary())

    ## another way to build the Sequential model:
    # model = models.Sequential()
    # model.add(layers.Flatten(input_shape=(28,28))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(10))

    ## loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = keras.optimizers.Adam(lr=0.001)
    metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optim,
                  metrics=metrics)

    ## training
    bs = 64 # batch_size
    epochs = 5

    model.fit(xtrain,
              ytrain,
              batch_size=bs,
              epochs=epochs,
              shuffle=True,
              verbose=2)

    ## evaulate
    model.evaluate(xtest,
                   ytest,
                   batch_size=bs,
                   verbose=2)

    ## predictions
    ## 1. option: build new model with Softmax layer
    probability_model = models.Sequential([model,
                                           layers.Softmax()])

    predictions = probability_model(xtest)
    pred0 = predictions[0]
    print(pred0)

    ## use np.argmax to get label with highest probability
    label0 = np.argmax(pred0)
    print(label0)

    ## 2. option: original model + nn.softmax, call model(x)
    # predictions = model(xtest)
    # predictions = tf.nn.softmax(predictions)
    # pred0 = predictions[0]
    # print(pred0)
    # label0 = np.argmax(pred0)
    # print(label0)

    ## 3. option: original model + nn.softmax, call model.predict(x)
    # predictions = model.predict(xtest, batch_size=bs)
    # predictions = tf.nn.softmax(predictions)
    # pred0 = predictions[0]
    # print(pred0)
    # label0 = np.argmax(pred0)
    # print(label0)

    ## call argmax for multiple labels
    # pred05s = predictions[0:5]
    # print(pred05s.shape)
    # label05s = np.argmax(pred05s, axis=1)
    # print(label05s)

if __name__ == "__main__":
    main()