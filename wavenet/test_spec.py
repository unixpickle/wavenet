"""
Tests for model specifications.
"""

from .spec import Layer, Network


def test_network_rf():
    """
    Test network receptive fields.
    """
    net = Network([Layer(i) for i in range(4)])
    assert net.receptive_field == 16

    net = Network([Layer(2), Layer(0), Layer(1)])
    assert net.receptive_field == 8
