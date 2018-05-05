"""
Abstract model specification.
"""


class Layer:
    """
    A description of a WaveNet layer.
    """

    def __init__(self, dilation, hidden_size=None):
        """
        Create a new Layer.

        Args:
          dilation: the base 2 logarithm of the dilation.
            0 means undilated, 1 means dilated by a factor
            of 2, etc.
          hidden_size: if specified, this is the number of
            channels in the dilated convolution output,
            before the channel-wise FC layer brings the
            number of channels back to that of the input.
            If None, the depth of the input is used.
        """
        self.dilation = dilation
        self.hidden_size = hidden_size

    @property
    def receptive_field(self):
        """
        Get the receptive field of the layer.

        This is the number of timesteps across the dilated
        filter spans. For example, if dilation == 3, then
        the filter spans 9 inputs, even though most of the
        inputs are ignored.
        """
        return 2 ** self.dilation + 1


class Network:
    """
    A description of a full WaveNet model.
    """

    def __init__(self, layers):
        """
        Create a Network.

        Args:
          layers: a sequence of layers, ordered from the
            input to the output.
        """
        self.layers = layers

    @property
    def receptive_field(self):
        """
        Compute the receptive field of the network.

        This reports the number of timesteps back the
        network is able to see. For example, if the
        network can see the current timestep and the two
        previous timesteps, this returns 3.
        """
        assert len(self.layers) > 0
        current_field = self.layers[0].receptive_field
        for layer in self.layers[1:]:
            # Subtract one because layer.receptive_field
            # already counts one of the same timesteps as
            # current_field.
            current_field += layer.receptive_field - 1
        return current_field
