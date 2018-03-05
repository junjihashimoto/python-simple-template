"""
>>> batch = 7
>>> num_seq = 5
>>> num_input = 11
>>> num_state = 128
>>> x = tf.zeros([batch,12],name="x")
>>> input = tf.zeros([batch, num_seq, num_input],name="input")
>>> input
<tf.Tensor 'input:0' shape=(7, 5, 11) dtype=float32>
>>> input0 = tf.zeros([batch, num_input],name="input0")
>>> input0
<tf.Tensor 'input0:0' shape=(7, 11) dtype=float32>
>>> state0 = tf.nn.rnn_cell.LSTMStateTuple(c=tf.zeros([batch, num_state],name="c"),h=tf.zeros([batch, num_state],name="h"))
>>> state0 
LSTMStateTuple(c=<tf.Tensor '...' shape=(7, 128) dtype=float32>, h=<tf.Tensor '...' shape=(7, 128) dtype=float32>)
>>> lstm = tf.nn.rnn_cell.LSTMCell(num_units=128)
>>> lstm(input0,state0)
(<tf.Tensor '...' shape=(7, 128) dtype=float32>, LSTMStateTuple(c=<tf.Tensor '...' shape=(7, 128) dtype=float32>, h=<tf.Tensor '...' shape=(7, 128) dtype=float32>))
>>> stacked_cells = []
>>> for i in range(2): stacked_cells.append(tf.nn.rnn_cell.LSTMCell(num_units=128))
>>> cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_cells)
>>> cell
<tensorflow.python.ops.rnn_cell_impl.MultiRNNCell object at ...>
>>> outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=input, dtype=tf.float32)
>>> outputs
<tf.Tensor '...' shape=(7, 5, 128) dtype=float32>
>>> states
(LSTMStateTuple(c=<tf.Tensor '...' shape=(7, 128) dtype=float32>, h=<tf.Tensor '...' shape=(7, 128) dtype=float32>), LSTMStateTuple(c=<tf.Tensor '...' shape=(7, 128) dtype=float32>, h=<tf.Tensor '...' shape=(7, 128) dtype=float32>))
"""
import numpy as np
import tensorflow as tf

