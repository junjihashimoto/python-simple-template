'''
>>> batch_size=3
>>> encoder_length=4
>>> decoder_length=5
>>> num_units=6
>>> src_vocab_size=7
>>> embedding_size=8
>>> tgt_vocab_size=9
>>> learning_rate = 0.01
>>> max_gradient_norm = 5.0
>>> beam_width =9
>>> use_attention = False
>>> tgt_sos_id = 7
>>> tgt_eos_id = 8
>>> encoder_inputs = tf.zeros([encoder_length, batch_size], tf.int32)
>>> encoder_inputs
<tf.Tensor '...' shape=(4, 3) dtype=int32>

>>> embedding_encoder = tf.get_variable("embedding_encoder", [src_vocab_size, embedding_size])
>>> embedding_encoder
<tf.Variable '...' shape=(7, 8) dtype=float32_ref>

>>> encoder_emb_inputs = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs)
>>> encoder_emb_inputs
<tf.Tensor '...' shape=(4, 3, 8) dtype=float32>

>>> encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
>>> encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inputs, time_major=True, dtype=tf.float32)
>>> encoder_outputs
<tf.Tensor '...' shape=(4, 3, 6) dtype=float32>
>>> encoder_state
LSTMStateTuple(c=<tf.Tensor '...' shape=(3, 6) dtype=float32>, h=<tf.Tensor '...' shape=(3, 6) dtype=float32>)

>>> decoder_inputs = tf.placeholder(tf.int32, shape=(decoder_length, batch_size), name="decoder_inputs")
>>> decoder_inputs
<tf.Tensor '...' shape=(5, 3) dtype=int32>

>>> decoder_lengths = tf.placeholder(tf.int32, shape=(batch_size), name="decoer_length")
>>> decoder_lengths
<tf.Tensor '...' shape=(3,) dtype=int32>

>>> embedding_decoder = tf.get_variable("embedding_decoder", [tgt_vocab_size, embedding_size])
>>> embedding_decoder
<tf.Variable '...' shape=(9, 8) dtype=float32_ref>

>>> decoder_emb_inputs = tf.nn.embedding_lookup(embedding_decoder, decoder_inputs)
>>> decoder_emb_inputs
<tf.Tensor '...' shape=(5, 3, 8) dtype=float32>

>>> projection_layer = tf.layers.Dense(tgt_vocab_size, use_bias=False)
>>> projection_layer
<tensorflow.python.layers.core.Dense object at ...>

>>> helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, decoder_lengths, time_major=True)
>>> helper
<tensorflow.contrib.seq2seq.python.ops.helper.TrainingHelper object at ...>

>>> decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
>>> decoder_cell
<tensorflow.python.ops.rnn_cell_impl.BasicLSTMCell object at ...>

>>> initial_state = encoder_state
>>> initial_state
LSTMStateTuple(c=<tf.Tensor '...' shape=(3, 6) dtype=float32>, h=<tf.Tensor '...' shape=(3, 6) dtype=float32>)

>>> decoder = tf.contrib.seq2seq.BasicDecoder(encoder_cell, helper, initial_state,output_layer=projection_layer)
>>> decoder
<tensorflow.contrib.seq2seq.python.ops.basic_decoder.BasicDecoder object at ...>

>>> final_outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
>>> final_outputs
BasicDecoderOutput(rnn_output=<tf.Tensor '...' shape=(3, ?, 9) dtype=float32>, sample_id=<tf.Tensor '...' shape=(3, ?) dtype=int32>)
>>> _final_state
LSTMStateTuple(c=<tf.Tensor '...' shape=(3, 6) dtype=float32>, h=<tf.Tensor '...' shape=(3, 6) dtype=float32>)
>>> _final_sequence_lengths 
<tf.Tensor '...' shape=(3,) dtype=int32>

'''
import numpy as np
import tensorflow as tf
