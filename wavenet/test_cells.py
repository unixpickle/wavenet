"""
Test RNN cells.
"""

import numpy as np
import tensorflow as tf

from .network import Conv, Network


def test_cell_equivalence():
    """
    Test that an RNNCell for a network gives the same
    output as the network itself.
    """
    with tf.Graph().as_default():
        net = Network([Conv(5, 2 ** i, dtype=tf.float64) for i in range(3)])
        assert net.receptive_field == 8
        in_seq = tf.concat([tf.random_normal([3, 9, 5], dtype=tf.float64)] * 2, axis=1)
        actual, _ = tf.nn.dynamic_rnn(net.cell(), in_seq, dtype=tf.float64)
        expected = net.apply(in_seq)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual, expected = sess.run((actual, expected))
            assert np.allclose(actual, expected)
