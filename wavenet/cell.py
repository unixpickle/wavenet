"""
RNN cells for WaveNet components.
"""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell  # pylint: disable=E0611

# Disable warning about "compute_output_shape" not being
# overridden, since most RNNCells don't seem to do so.
# pylint: disable=W0223


class ConvCell(RNNCell):
    """
    A recurrent cell that applies a dilated convolution
    using a caching mechanism to prevent re-computation.
    """

    def __init__(self, conv):
        """
        Create a ConvCell from a Conv instance.
        """
        super(ConvCell, self).__init__()
        self.conv = conv

    @property
    def state_size(self):
        return (tf.TensorShape([self.conv.channels]),) * (self.conv.receptive_field - 1)

    @property
    def output_size(self):
        return self.conv.channels

    def zero_state(self, batch_size, dtype):
        """
        Generate an all-zero cache for the cell.
        """
        zeros = tf.zeros([batch_size, self.conv.channels], dtype=dtype)
        return (zeros,) * (self.conv.receptive_field - 1)

    def call(self, inputs, state):  # pylint: disable=W0221
        old_inputs = state[0]
        new_cache = state[1:] + (inputs,)
        outputs = self.conv.apply_once(old_inputs, inputs)
        return outputs, new_cache
