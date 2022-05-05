import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
keras = tf.keras
layers = keras.layers
# preprocessing = layers.experimental.preprocessing
models = keras.models
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.set_printoptions(precision=3, suppress=True)
