import tensorflow as tf
from tf_flags import FLAGS

def distort_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string),
    })

    image = tf.decode_raw(features['image_raw'], tf.float32)
    label = tf.decode_raw(features['label_raw'], tf.float32)
    image = tf.reshape(image, [112, 112, FLAGS.nchannel])
    image.set_shape([112, 112, FLAGS.nchannel])
    label.set_shape([FLAGS.num_classes])
    print(image.get_shape())
    print(label.get_shape())
    #image.set_shape([FLAGS.crop_height, FLAGS.crop_width, FLAGS.nchannel])
    return image, label

def inputs(filenames, batch_size, num_epochs, num_threads, num_examples_per_epoch, shuffle=True, distort=False):
    if not num_epochs:
        num_epochs = None
    for filename in filenames:
        if not tf.gfile.Exists(filename):
            raise ValueError('Failed to find file: ' + filename)
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=shuffle, name='string_input_producer'
        )
        image, label = read_and_decode(filename_queue)
        if distort:
            image = distort_image(image)
        print('Image shape is ', image.get_shape())
        print('Label shape is ', label.get_shape())
        min_fraction_of_examples_in_queue = 0.5
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
        if shuffle:
            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                min_after_dequeue=min_queue_examples,
                name='batching_shuffling'
            )
        else:
            images, labels = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=num_examples_per_epoch,
                enqueue_many=False,
                allow_smaller_final_batch=False,
                name='batching_shuffling'
            )
        print('Images shape is ', images.get_shape())
        print('Labels shape is ', labels.get_shape())
        print('######################################################################')

    return images, labels
