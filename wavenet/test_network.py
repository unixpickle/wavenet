"""
Tests for networks.
"""

import tensorflow as tf

from .network import Conv, Network


def test_network_rf():
    """
    Test network receptive fields.
    """
    with tf.Graph().as_default():
        net = Network([Conv(7, i) for i in range(4)])
        assert net.receptive_field == 16

        net = Network([Conv(7, 2), Conv(7, 0), Conv(7, 1)])
        assert net.receptive_field == 8

        net = Network([Conv(7, 2), Conv(7, 0), Conv(7, 1), Conv(7, 0)])
        assert net.receptive_field == 9
