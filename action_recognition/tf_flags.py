import tensorflow as tf
import datetime

flags = tf.app.flags

log_dir = 'tensorboard_events/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
# Basic model parameters as external flags.
flags = tf.app.flags
#flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
flags.DEFINE_integer('num_examples_per_epoch', 1000, 'Number of examples to go through for each epoch')
flags.DEFINE_string('checkpoint_dir', 'checkpoint/', 'Directory where to read/write model checkpoints')
flags.DEFINE_string('pretrained_model', 'pretrained/conv3d_deepnetA_sport1m_iter_1900000_TF.model', 'The pretrained model on ucf101')
flags.DEFINE_string('training_data', 'data/train', 'Directory to the tfrecords files for training')
flags.DEFINE_string('validation_data', 'data/val', 'Directory to the tfrecords files for validation')
flags.DEFINE_string('log_dir', log_dir, 'Directory where to write the tensorboard events')
flags.DEFINE_integer('max_steps', 50000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 32, 'Batch size.')
flags.DEFINE_integer('num_threads', 10, 'Number of threads enqueuing tensor list')
flags.DEFINE_integer('display_step', 1, 'Number of steps for disply')
flags.DEFINE_integer('val_step', 10, 'Number of steps for validation')
flags.DEFINE_integer('save_step', 10, 'Number of step to save the model')
flags.DEFINE_integer('num_classes', 101, 'Number of classes to do classification')
flags.DEFINE_string('tower_name', 'tower', 'Name of the tower')
flags.DEFINE_integer('gpu_num', 2, 'Number of gpus to run')
flags.DEFINE_integer('depth', 16, 'Depth for the video')
flags.DEFINE_integer('width', 160, 'Width for the video')
flags.DEFINE_integer('height', 120, 'Height for the video')
flags.DEFINE_integer('nchannel', 3 ,'Number of channel for the video')
flags.DEFINE_boolean('normalize', False, 'Whether to normalize to [0, 1] range')
FLAGS = flags.FLAGS
