# nd-array
# GPU support
# build computational graph / backpropagation
# immutable

from turtle import st
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main():

    # x = tf.constant(4, shape=(1,1), dtype=tf.float32)

    # rank-1
    # x = tf.constant([1,2,3])

    # rank-2
    # x = tf.constant([[1,2,3], [4,5,6]])

    # 3x3
    # x = tf.ones((3,3))
    # x = tf.zeros((3,3))
    # x = tf.eye(3)
    # x = tf.random.normal((3,3), mean=0, stddev=1)
    # x = tf.random.uniform((3,3), minval=0, maxval=1)
    # x = tf.range(10)

    # cast
    # x = tf.cast(x, dtype=tf.float32)
    # print(x)

    # elementwise
    x = tf.constant([1,2,3])
    y = tf.constant([4,5,6])
    z = tf.add(x,y)
    print(z)

if __name__ == "__main__":
    main()