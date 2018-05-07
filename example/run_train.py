"""
Train a WaveNet model.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os

import tensorflow as tf

from data import dir_dataset, save_audio
from model import Model


def main():
    args = arg_parser().parse_args()
    print('Building graph...')
    model = Model()
    train_data, val_data = dir_dataset(args.data_dir,
                                       max_secs=args.max_secs,
                                       sample_rate=args.sample_rate)
    train_batch = train_data.batch(args.batch_size).repeat().make_one_shot_iterator().get_next()
    val_batch = val_data.batch(args.batch_size).repeat().make_one_shot_iterator().get_next()

    train_loss = tf.reduce_mean(model.log_loss(train_batch))
    val_loss = tf.reduce_mean(model.log_loss(val_batch))

    samples = model.sample(args.sample_batch, int(args.max_secs * args.sample_rate))
    write_samples = save_audio(args.sample_path, samples, sample_rate=args.sample_rate)
    optimize = tf.train.AdamOptimizer(learning_rate=args.step_size).minimize(train_loss)

    global_step = tf.get_variable('global_step',
                                  initializer=tf.constant(0, dtype=tf.int32),
                                  trainable=False)
    inc_step = tf.assign_add(global_step, tf.constant(1, dtype=tf.int32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if tf.train.latest_checkpoint(args.checkpoint):
            print('Restoring from checkpoint...')
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
        print('Training...')
        while True:
            step, tl, vl, _ = sess.run([inc_step, train_loss, val_loss, optimize])
            print('step %d: train=%f val=%f' % (step, tl, vl))
            if step % args.sample_interval == 0:
                sess.run(write_samples)
            if step % args.save_interval == 0:
                saver.save(sess, os.path.join(args.checkpoint, 'ckpt'), global_step=global_step)


def arg_parser():
    """
    Get an ArgumentParser for the CLI arguments.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-dir', help='data directory of WAV files', default='data')
    parser.add_argument('--max-secs', help='maximum time in seconds', type=float, default=3.0)
    parser.add_argument('--sample-rate', help='audio sample rate', type=int, default=22050)
    parser.add_argument('--checkpoint', help='checkpoint directory', default='checkpoint')
    parser.add_argument('--save-interval', help='iters per save', type=int, default=1000)
    parser.add_argument('--sample-interval', help='how often to save a sample',
                        type=int, default=10000)
    parser.add_argument('--sample-batch', help='how many samples to generate',
                        type=int, default=16)
    parser.add_argument('--sample-path', help='where to dump the latest sample',
                        default='sample.wav')
    parser.add_argument('--step-size', help='training step size', type=float, default=1e-3)
    parser.add_argument('--batch-size', help='SGD batch size', type=int, default=1)
    return parser


if __name__ == '__main__':
    main()
