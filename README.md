# wavenet

This is a Python module for [WaveNet](https://arxiv.org/abs/1609.03499) in TensorFlow. It implements both a traditional convolutional API, and an `RNNCell` for fast stepping.

# Usage

You can install the package with `pip` like so:

```
$ pip install wavenet
```

The [example](example) is a complete program which trains a WaveNet on synthetic data from the macOS speech synthesizer. In this README, we only show the basics of applying WaveNets to sequences.

Once you have the `wavenet` package, it's trivial to construct and use a WaveNet. Here's an example:

```python
import tensorflow as tf
from wavenet import Conv, Network

# Produce a [batch x timesteps x depth] input sequence.
# Only `depth` needs to be known ahead of time.
inputs = ...

# Create a new network and all its variables.
# In this case, the receptive field is 16.
num_channels = inputs.get_shape()[-1].value
network = Network([Conv(channels=num_channels, dilation=2**i) for i in range(4)])

# Apply the model to the inputs, yielding outputs for
# every timestep of the input.
outputs = network.apply(inputs)
```

If you need to step the model efficiently, or if you want to use a WaveNet in place of another recurrent neural network, you can get an `RNNCell` directly from a `Network` by calling `network.cell()`. In the above example, you could replace `network.apply(inputs)` with this code that uses an `RNNCell`:

```python
rnn_cell = network.cell()
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype=tf.float32)
```
