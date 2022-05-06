import os
import math
import random
import shutil

import numpy as np
import pandas as pd

import tensorflow as tf

keras = tf.keras
layers = keras.layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def reorganize():
    # download the data from kaggle:
    # https://www.kaggle.com/ihelon/lego-minifigures-tensorflow-tutorial
    # move the folder into your project folder and create a backup of
    # the star-wars images at 'lego/star-wars-images/'
    BASE_DIR = 'lego/star-wars-images/'
    names = ["YODA", "LUKE SKYWALKER", "R2-D2", "MACE WINDU", "GENERAL GRIEVOUS"]

    tf.random.set_seed(1)
    # Reorganize the folder structure:
    if not os.path.isdir(BASE_DIR + 'train/'):
        for name in names:
            os.makedirs(BASE_DIR + 'train/' + name)
            os.makedirs(BASE_DIR + 'val/' + name)
            os.makedirs(BASE_DIR + 'test/' + name)

    # Move the image files
    orig_folders = ["0001/", "0002/", "0003/", "0004/", "0005/"]
    for folder_idx, folder in enumerate(orig_folders):
        files = os.listdir(BASE_DIR + folder)
        number_of_images = len([name for name in files])
        n_train = int((number_of_images * 0.6) + 0.5)
        n_valid = int((number_of_images*0.25) + 0.5)
        n_test = number_of_images - n_train - n_valid
        print(number_of_images, n_train, n_valid, n_test)
        for idx, file in enumerate(files):
            file_name = BASE_DIR + folder + file
            if idx < n_train:
                shutil.move(file_name, BASE_DIR + "train/" + names[folder_idx])
            elif idx < n_train + n_valid:
                shutil.move(file_name, BASE_DIR + "val/" + names[folder_idx])
            else:
                shutil.move(file_name, BASE_DIR + "test/" + names[folder_idx])

def preprocessing():

    # Generate batches of tensor image data with
    # optional real-time data augmentation.
    # preprocessing_function
    # rescale=1./255 -> [0,1]
    train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    #    rotation_range=20,
    #    horizontal_flip=True,
    #    width_shift_range=0.2, height_shift_range=0.2,
    #    shear_range=0.2, zoom_range=0.2)

    valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_batches = train_gen.flow_from_directory('lego/star-wars-images/train',
                                                  target_size=(256, 256),
                                                  class_mode='sparse',
                                                  batch_size=4,
                                                  shuffle=True,
                                                  color_mode="rgb",
                                                  classes=names)

    val_batches = valid_gen.flow_from_directory('lego/star-wars-images/val',
                                                target_size=(256, 256),
                                                class_mode='sparse',
                                                batch_size=4,
                                                shuffle=False,
                                                color_mode="rgb",
                                                classes=names)

    test_batches = test_gen.flow_from_directory('lego/star-wars-images/test',
                                                target_size=(256, 256),
                                                class_mode='sparse',
                                                batch_size=4,
                                                shuffle=False,
                                                color_mode="rgb",
                                                classes=names)
    train_batch = train_batches[0]
    print(train_batch[0].shape)
    print(train_batch[1])
    test_batch = test_batches[0]
    print(test_batch[0].shape)
    print(test_batch[1])

    return train_batches, val_batches, test_batches

def create_model():

    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, (3,3),
                            strides=(1,1),
                            padding="valid",
                            activation='relu',
                            input_shape=(256, 256,3)))

    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5))
    print(model.summary())

    return model

def loss_and_optimize():

    # loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = keras.optimizers.Adam(lr=0.001)
    metrics = ["accuracy"]

    model.compile(optimizer=optim, loss=loss, metrics=metrics)

    return model

def train():

    # training
    epochs = 30

    # callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=5,
                                                   verbose=2)

    history = model.fit(train_batches, validation_data=val_batches,
                        callbacks=[early_stopping],
                        epochs=epochs, verbose=2)

    model.save("lego_model.h5")

    return history

def evaluate():

     # evaluate on test data
    model.evaluate(test_batches, verbose=2)

    # make some predictions
    predictions = model.predict(test_batches)
    predictions = tf.nn.softmax(predictions)
    labels = np.argmax(predictions, axis=1)

    print(test_batches[0][1])
    print(labels[0:4])


def transfer_learning():

    ##  Transfer Learning
    vgg_model = tf.keras.applications.vgg16.VGG16()
    print(type(vgg_model))
    vgg_model.summary()

    # try out different ones, e.g. MobileNetV2
    #tl_model = tf.keras.applications.MobileNetV2()
    #print(type(tl_model))
    #tl_model.summary()
    # convert to Sequential model, omit the last layer
    # this works with VGG16 because the structure is linear
    model = keras.models.Sequential()
    for layer in vgg_model.layers[0:-1]:
        model.add(layer)
    model.summary()

    # set trainable=False for all layers
    # we don't want to train them again
    for layer in model.layers:
        layer.trainable = False
    model.summary()

    # add a last classification layer for our use case with 5 classes
    model.add(layers.Dense(5))

    # loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = keras.optimizers.Adam(lr=0.001)
    metrics = ["accuracy"]

    model.compile(optimizer=optim, loss=loss, metrics=metrics)

    # get the preprocessing function of this model
    preprocess_input = tf.keras.applications.vgg16.preprocess_input

    # Generate batches of tensor image data with real-time data augmentation.

    train_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

    train_batches = train_gen.flow_from_directory('lego/star-wars-images/train',
                                                  target_size=(224, 224),
                                                  class_mode='sparse',
                                                  batch_size=4,
                                                  shuffle=True,
                                                  color_mode="rgb",
                                                  classes=names)

    val_batches = valid_gen.flow_from_directory('lego/star-wars-images/val',
                                                target_size=(224, 224),
                                                class_mode='sparse',
                                                batch_size=4,
                                                shuffle=True,
                                                color_mode="rgb",
                                                classes=names)

    test_batches = test_gen.flow_from_directory('lego/star-wars-images/test',
                                                target_size=(224, 224),
                                                class_mode='sparse',
                                                batch_size=4,
                                                shuffle=False,
                                                color_mode="rgb",
                                                classes=names)

    epochs = 30

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=5,
                                                   verbose=2)

    model.fit(train_batches, validation_data=val_batches,
            callbacks=[early_stopping],
            epochs=epochs, verbose=2)

    model.evaluate(test_batches, verbose=2)

def main():

    reorganize()

    train_batches, val_batches, test_batches = preprocessing()

    model = create_model()

    model = loss_and_optimize()

    history = train()

    evaluate()

    transfer_learning()
