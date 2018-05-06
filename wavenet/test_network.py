"""
Tests for networks.
"""

import numpy as np
import tensorflow as tf

from .network import Conv, Network


def test_network_rf():
    """
    Test network receptive fields.
    """
    with tf.Graph().as_default():
        net = Network([Conv(7, 2 ** i) for i in range(4)])
        assert net.receptive_field == 16

        net = Network([Conv(7, 4), Conv(7, 1), Conv(7, 2)])
        assert net.receptive_field == 8

        net = Network([Conv(7, 4), Conv(7, 1), Conv(7, 2), Conv(7, 1)])
        assert net.receptive_field == 9


def test_network_apply():
    """
    Test applying a WaveNet model.

    This is mostly intended to catch runtime errors.
    """
    with tf.Graph().as_default():
        net = Network([Conv(5, 2 ** i, dtype=tf.float64) for i in range(3)])
        assert net.receptive_field == 8
        # A sequence that repeats at an interval that is
        # greater than the receptive field.
        in_seq = tf.concat([tf.random_normal([3, 9, 5], dtype=tf.float64)] * 2, axis=1)
        out_seq = net.apply(in_seq)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs = sess.run(out_seq)
            # The same, because receptive fields are identical.
            assert np.allclose(outputs[:, 8], outputs[:, 17])
            # Not the same because of being different timesteps.
            assert not np.allclose(outputs[:, 8], outputs[:, 16])
            # Not the same because of zero padding.
            assert not np.allclose(outputs[:, 0], outputs[:, 9])
