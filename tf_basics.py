# nd-array
# GPU support
# build computational graph / backpropagation
# immutable

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
    z = x+y

    z = tf.subtract(x, y)
    z = x+y

    z = tf.divide(x, y)
    z = x/y

    z = tf.multiply(x, y)
    z = x*y

    z = tf.tensordot(x, y, axis=1)

    z = x ** 3

    x = tf.random.normal((2,2))
    y = tf.random.normal((2,2))

    x = tf.random.normal((2,3))
    y = tf.random.normal((3,2))
    x = tf.random.normal((2,3))
    y = tf.random.normal((3,4))
    z = tf.matmul(x,y)
    z = x @ y
    print(z)

    # slicing, indexing
    # same as numpy, python

    # reshaping
    x = tf.random.normal((2,3))
    # print(x)

    x = tf.reshape(x, (3,2))
    # print(x)

    x = tf.reshape(x, (-1,2))
    # print(x)

    x = tf.reshape(x, (6))
    # print(x)

    # numpy
    x = x.numpy()
    # print(type(x))

    x = tf.convert_to_tensor(x)
    # print(type(x))
    # -> eager tensor = evaluates operations immediately
    # without building graphs

    # string tensor
    x = tf.constant("Patrick")
    print(x)

    x = tf.constant(["Patrick", "Max", "Mary"])
    print(x)

    # Variable
    # A tf.Variable represents a tensor whose value can be
    # changed by running ops on it
    # Used to represent shared, persistent state your program manipulates
    # Higher level libraries like tf.keras use tf.Variable to store model parameters.
    b = tf.Variable([[1.0, 2.0, 3.0]])
    print(b)
    print(type(b))

if __name__ == "__main__":
    main()