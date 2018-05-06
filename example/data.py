"""
Datasets of audio samples.
"""

from hashlib import md5
import os

import tensorflow as tf


def save_audio(out_path, tensor, sample_rate=22050):
    """
    Write a 1-D audio Tensor to a WAV file.
    """
    shaped = tf.reshape(tensor, [-1, 1])
    # pylint: disable=E1101
    data = tf.contrib.ffmpeg.encode_audio(shaped, file_format='wav', samples_per_second=sample_rate)
    return tf.write_file(out_path, data)


def dir_dataset(dir_path, max_secs=3.0, sample_rate=22050):
    """
    Create a dataset of audio clips from a directory.

    Args:
      dir_path: a directory full of .wav files.
      max_secs: the duration for every clip to be padded
        or cropped to.
      sample_rate: the audio samples per second.

    Returns:
      A tuple (train, validation) of datasets of audio
        clips, where each clip is a 1-D Tensor of float32
        samples in the range [-1, 1].
    """
    paths = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if x.endswith('.wav')]
    train_paths = [p for p in paths if not _use_for_val(p)]
    val_paths = [p for p in paths if _use_for_val(p)]
    return tuple(dataset_from_paths(paths, max_secs, sample_rate)
                 for paths in [train_paths, val_paths])


def dataset_from_paths(paths, max_secs, sample_rate):
    """
    Create a Dataset from the audio file paths.
    """
    num_samples = int(max_secs * sample_rate)
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)

    def read_clip(path_tensor):
        data_tensor = tf.read_file(path_tensor)
        # pylint: disable=E1101
        audio_tensor = tf.contrib.ffmpeg.decode_audio(data_tensor,
                                                      file_format='wav',
                                                      samples_per_second=sample_rate,
                                                      channel_count=1)
        audio_tensor = tf.reshape(audio_tensor, [-1])
        return pad_or_crop(audio_tensor, num_samples)

    return paths_ds.shuffle(buffer_size=len(paths)).map(read_clip)


def pad_or_crop(tensor, length):
    """
    Pad or crop a 1-D tensor to be a certain length.
    """
    cur_len = tf.shape(tensor)[0]
    return tf.cond(cur_len < length,
                   true_fn=lambda: tf.pad(tensor, [[0, length - cur_len]]),
                   false_fn=lambda: tensor[:length])


def _use_for_val(path):
    return md5(bytes(path, 'utf-8')).digest()[0] < 0x80
