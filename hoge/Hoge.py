"""
>>> 3+3
6

"""

import tensorflow as tf
import random


def prop_int_eq(fn):
    """
    >>> prop_int_eq(lambda x: (1 <= x and x <= 100))
    True
    """
    num_dim = 10
    max_dim = 100
    min_dim = 1
    random_dim = list(map(lambda x: random.randint(min_dim, max_dim), range(1, num_dim) ))
    result_list = list(map(lambda x: fn(x),random_dim))
    return all(result_list)

def shape_eq(x,out_shape):
    return x.shape == tf.TensorShape(out_shape)

def type_eq(x,out_dtype):
    return x.dtype == out_dtype

def ttype_eq(x,out_shape,out_dtype):
    return shape_eq(x,out_shape) and type_eq(x,out_dtype)

def inout_eq(fn,in_shape,in_dtype,out_shape,out_dtype):
    v = fn(tf.zeros(in_shape,dtype=in_dtype))
    r = ttype_eq(v,out_shape,out_dtype)
    if not r:
        print("in_shape:")
        print(in_shape)
        print("expected out_shape:")
        print(out_shape)
        print("got      out_shape:")
        print(v.shape)
    return r


def myreshape(x):
    """
    >>> x = tf.zeros([10,56,56,1],name='x')
    >>> myreshape(x)
    <tf.Tensor 're:0' shape=(40, 28, 28, 1) dtype=float32>
    >>> v = myreshape(x)
    >>> shape_eq(v,[40,28,28,1])
    True
    >>> type_eq(v,tf.float32)
    True
    >>> ttype_eq(v,[40,28,28,1],tf.float32)
    True
    >>> ttype_eq(v,[40,28,28,1],tf.float32)
    True
    >>> inout_eq(myreshape,\
                 in_shape=[10,56,56,1],in_dtype=tf.float32,\
                 out_shape=[40,28,28,1],out_dtype=tf.float32)
    True
    >>> prop_int_eq(lambda n: inout_eq(myreshape,\
                                       in_shape=[n,56,56,1],in_dtype=tf.float32,\
                                       out_shape=[n*4,28,28,1],out_dtype=tf.float32))
    True
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
