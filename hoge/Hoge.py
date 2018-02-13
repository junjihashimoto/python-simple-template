"""
>>> 3+3
6

"""

import tensorflow as tf

def myreshape(x):
    """
    >>> x = tf.zeros([10,56,56,1],name='x')
    >>> myreshape(x)
    <tf.Tensor 're:0' shape=(40, 28, 28, 1) dtype=float32>
    
    """
    return tf.reshape(x, [-1, 28, 28, 1], name="re")


class Piyo:
    def __init__(self):
        self.name = ""

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def one(self):
        return 1

    def two(self):
        return 'two'
