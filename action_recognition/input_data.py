import tensorflow as tf
import numpy as np
from tf_flags import FLAGS

#np_mean = np.load('crop_mean.npy').reshape([16, 112, 112, 3])

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'video_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })

    video = tf.decode_raw(features['video_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    video = tf.reshape(video, [FLAGS.depth, FLAGS.height, FLAGS.width, FLAGS.nchannel])
    video = tf.random_crop(video, [16, 112, 112, 3])
    #video = tf.cast(video, tf.float32)* (1.0 / 255)
    video = tf.cast(video, tf.float32)
    #video = video - np_mean
    if FLAGS.normalize:
        video = tf.cast(video, tf.float32)
        video = tf.cast(video, tf.float32) * (1.0 / 255) - 0.5
    return video, label

def inputs(filenames, batch_size, num_epochs, num_threads, num_examples_per_epoch, shuffle=True):
    if not num_epochs:
        num_epochs = None
    for filename in filenames:
        if not tf.gfile.Exists(filename):
            raise ValueError('Failed to find file: ' + filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=shuffle, name='string_input_producer'
        )
        video, label = read_and_decode(filename_queue, normalize=False)

        print('Video shape is ', video.get_shape())
        print('Label shape is ', label.get_shape())
        min_fraction_of_examples_in_queue = 0.5
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        if shuffle:
            videos, sparse_labels = tf.train.shuffle_batch(
                [video, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                min_after_dequeue=min_queue_examples,
                name='batching_shuffling'
            )
        else:
            videos, sparse_labels = tf.train.batch(
                [video, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                allow_smaller_final_batch=False,
                min_after_dequeue=min_queue_examples,
                name='batching_shuffling'
            )
        print('Videos shape is ', videos.get_shape())
        print('Label shape is ', sparse_labels.get_shape())
        print('######################################################################')

    return videos, sparse_labels
