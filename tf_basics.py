# nd-array
# GPU support
# build computational graph / backpropagation
# immutable

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# x = tf.constant(4, shape=(1,1), dtype=tf.float32)

# rank-1
# x = tf.constant([1,2,3])

# rank-2
# x = tf.constant([[1,2,3], [4,5,6]])

#3x3
# x = tf.ones((3,3))
x = tf.zeros((3,3))
x = tf.eye(3)
print(x)

