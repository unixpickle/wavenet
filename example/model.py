"""
Training & sampling for the WaveNet model.
"""

from math import log

import tensorflow as tf
from wavenet import Network, Conv


class Model:
    """
    An instance of a WaveNet model.
    """

    def __init__(self, feature_size=256, scope='wavenet'):
        self.feature_size = feature_size
        with tf.variable_scope(scope):
            self.wavenet = Network([Conv(feature_size, 2 ** i) for i in range(13)])

    def log_loss(self, samples):
        """
        Apply the WaveNet to a batch of continuous audio
        samples and get the log probabilities for every
        output.

        Args:
          samples: a [batch x timesteps] Tensor of float32
            audio samples.

        Returns:
          A Tensor of negative log probs, one per sample.
        """
        discrete = discretize_samples(samples)
        zero_embedding = tf.zeros([tf.shape(samples)[0], 1, self.feature_size])
        embedded = tf.concat([zero_embedding, self._embed_samples(discrete[:, 1:])], axis=1)
        wavenet_out = self.wavenet.apply(embedded)
        logits = self._output_logits(wavenet_out)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=discrete, logits=logits)

    def sample(self, batch_size, timesteps):
        """
        Produce a batch of samples from the model.
        """
        cell = self.wavenet.cell()

        def loop_condition(timestep, _states, _last_output, _arr):
            return timestep < timesteps

        def loop_body(timestep, states, last_output, arr):
            embedded = tf.cond(timestep > 0,
                               true_fn=lambda: self._embed_samples(last_output),
                               false_fn=lambda: tf.zeros([batch_size, self.feature_size]))
            wavenet_outs, new_states = cell(embedded, states)
            logits = self._output_logits(wavenet_outs)
            outputs = sample_logits(logits)
            return timestep + 1, new_states, outputs, arr.write(timestep, outputs)

        res = tf.while_loop(loop_condition,
                            loop_body,
                            [tf.constant(0, dtype=tf.int32),
                             cell.zero_state(batch_size, tf.float32),
                             tf.zeros([batch_size], tf.int32),
                             tf.TensorArray(tf.int32, size=timesteps)])

        return undiscretize_samples(res[-1].concat())

    def _embed_samples(self, discrete):
        """
        Embed the discretized samples.
        """
        embeddings = tf.get_variable('embeddings',
                                     shape=[256, self.feature_size],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer())
        return tf.gather(embeddings, tf.cast(discrete, tf.int32))

    def _output_logits(self, wavenet_out):
        """
        Turn wavenet outputs into logits.
        """
        out = tf.nn.relu(wavenet_out)
        out = tf.layers.dense(out, self.feature_size, activation=tf.nn.relu)
        out = tf.layers.dense(out, 256)
        return out


def discretize_samples(samples, mu=255):
    """
    Discretize continuous samples using the µ-law.
    """
    squeezed = tf.sign(samples) * (tf.log(1 + mu * tf.abs(samples)) / log(1 + mu))
    discrete = tf.cast((squeezed + 1) * 128 * (1 - 1e-4), tf.int32)
    return discrete


def undiscretize_samples(samples, mu=255):
    """
    Perform the inverse of discretize_samples().
    """
    continuous = tf.cast(samples, tf.float32) / 127.5 - 1
    unsqueezed = tf.sign(continuous) * (1 / mu) * (tf.pow(float(1 + mu), tf.abs(continuous)) - 1)
    return unsqueezed


def sample_logits(logits):
    """
    Sample integer values from a Tensor where the last
    dimension is a logit vector.
    """
    return tf.distributions.Categorical(logits=logits, dtype=tf.int32).sample()
