import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

keras = tf.keras
layers = keras.layers
models = keras.models
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(precision=3, suppress=True)

def main():
    # https://archive.ics.uci.edu/ml/datasets/Auto+MPG
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    dataset = pd.read_csv(url,
                          names=column_names,
                          na_values='?',
                          comment='\t',
                          sep=' ',
                          skipinitialspace=True)


    # clean data
    dataset = dataset.dropna()

    # convert categorical 'Origin' data into one-hot data
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1)*1
    dataset['Europe'] = (origin == 2)*1
    dataset['Japan'] = (origin == 3)*1

    # Split the data into train and test
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    print(dataset.shape, train_dataset.shape, test_dataset.shape)
    train_dataset.describe().transpose()


    # split features from labels
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    # Normalize
    print(train_dataset.describe().transpose()[['mean', 'std']])

    # Normalization
    normalizer = keras.utils.normalize(train_features)

    # When the layer is called it returns the input data, with each feature independently normalized:
    # (input-mean)/stddev
    first = np.array(train_features[:1])
    print('First example:', first)

    # Regression
    # 1. Normalize the input horsepower
    # 2. Apply a linear transformation (y = m*x+b) to produce 1 output using layers.Dense

    feature = 'Horsepower'
    single_feature = np.array(train_features[feature])
    print(single_feature.shape, train_features.shape)

    # Normalization
    single_feature_normalizer = keras.utils.normalize(single_feature)


    # Sequential model
    single_feature_model = keras.models.Sequential([single_feature_normalizer,
                                                    layers.Dense(units=1)]) # Linear Model

    single_feature_model.summary()

    # loss and optimizer
    loss = keras.losses.MeanAbsoluteError() # MeanSquaredError
    optim = keras.optimizers.Adam(lr=0.1)

    single_feature_model.compile(optimizer=optim, loss=loss)

    history = single_feature_model.fit(train_features[feature],
                                       train_labels,
                                       epochs=100,
                                       verbose=1,
                                       validation_split = 0.2) # Calculate validation results on 20% of the training data

    single_feature_model.evaluate(test_features[feature],
                                  test_labels,
                                  verbose=1)

    # DNN
    dnn_model = keras.Sequential([single_feature_normalizer,
                                  layers.Dense(64, activation='relu'),
                                  layers.Dense(64, activation='relu'),
                                  layers.Dense(1)])

    dnn_model.compile(loss=loss,
                      optimizer=tf.keras.optimizers.Adam(0.001))

    dnn_model.summary()

    dnn_model.fit(train_features[feature],
                  train_labels,
                  validation_split=0.2,
                  verbose=1,
                  epochs=100)

    dnn_model.evaluate(test_features[feature], test_labels, verbose=1)

    # multiple inputs
    linear_model = tf.keras.Sequential([normalizer,
                                        layers.Dense(units=1)])

    linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
                         loss=loss)

    linear_model.fit(train_features,
                     train_labels,
                     epochs=100,
                     verbose=1,
                     validation_split = 0.2)# Calculate validation results on 20% of the training data

    linear_model.evaluate(test_features,
                          test_labels,
                          verbose=1)

if __name__ == "__main__":
    main()