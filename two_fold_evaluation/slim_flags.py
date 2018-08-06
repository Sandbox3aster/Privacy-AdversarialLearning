import tensorflow as tf
import datetime
import os

flags = tf.app.flags


_lambda = 0.5
_mode = 'SuppressingMostConfident'
N = 3

resampling = False

isResampling = lambda bool: "Resample_" if bool else "NoResample_"

# Basic model parameters as external flags.
flags.DEFINE_string('ckpt_dir', '../DAN/checkpoint/17_attributes_' + isResampling(resampling) + '{}_'.format(N) + '{}'.format(_lambda), 'Directory where to read/write model checkpoints')

flags.DEFINE_string('train_videos_files_dir', '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/ucf101_tfrecords/train', 'The directory of the training files')
flags.DEFINE_string('val_videos_files_dir', '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/ucf101_tfrecords/val', 'The directory of the validation files')
#flags.DEFINE_string('test_videos_files_dir', '/home/wuzhenyu_sjtu/DAN_vispr/data/RGB/clean/test', 'The directory of the testing files')

flags.DEFINE_string('train_images_files_dir', '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/vispr_tfrecords/17attributes/train', 'The directory of the training files')
flags.DEFINE_string('val_images_files_dir', '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/vispr_tfrecords/17attributes/val', 'The directory of the validation files')
flags.DEFINE_string('test_images_files_dir', '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/vispr_tfrecords/17attributes/test', 'The directory of the testing files')


flags.DEFINE_integer('max_steps', 1500, 'Number of steps to run trainer.')
flags.DEFINE_integer('pretraining_steps_utility', 1000, 'Number of step for the pretraining')
flags.DEFINE_integer('pretraining_steps_budget', 200, 'Number of step for the pretraining')

flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('video_batch_size', 8, 'Batch size.')
flags.DEFINE_integer('image_batch_size', 128, 'Batch size.')
flags.DEFINE_integer('num_examples_per_epoch', 200, 'Epoch size')

flags.DEFINE_integer('num_threads', 10, 'Number of threads enqueuing tensor list')
flags.DEFINE_integer('val_step', 20, 'Number of steps for validation')
flags.DEFINE_integer('save_step', 5, 'Number of step to save the model')
flags.DEFINE_integer('num_classes_utility', 101, 'Number of classes to do classification')
flags.DEFINE_integer('num_classes_budget', 17, 'Number of classes to do classification')

flags.DEFINE_integer('gpu_num', 2, 'Number of gpus to run')
flags.DEFINE_integer('depth', 16, 'Depth for the video')
flags.DEFINE_integer('width', 160, 'Width for the video')
flags.DEFINE_integer('height', 120, 'Height for the video')
flags.DEFINE_integer('crop_width', 112, 'Width for the video')
flags.DEFINE_integer('crop_height', 112, 'Height for the video')

flags.DEFINE_integer('NBudget', N, 'Number of budget models')
flags.DEFINE_boolean('use_resampling', resampling, 'Whether to use resampling model')

flags.DEFINE_integer('nchannel', 3 ,'Number of channel for the video')
flags.DEFINE_boolean('use_crop', True, 'Whether to use crop when reading video in the input pipeline')
flags.DEFINE_boolean('use_random_crop', False, 'Whether to use random crop when reading video in the input pipeline')
flags.DEFINE_boolean('use_center_crop', True, 'Whether to use center crop when reading video in the input pipeline')
flags.DEFINE_boolean('use_frame_distortion', False, 'Whether to user frame distortion for data augmentation on video level')
flags.DEFINE_boolean('use_normalization', False, 'Whether to convert to [0,1] range')
flags.DEFINE_boolean('add_gaussian_noise', False, 'Whether to add Gaussian noise as data augmentation')
flags.DEFINE_integer('n_minibatches', 2, 'Number of mini-batches')

module_name = 'resnet_v1_50'

ckpt_path_map = {
                'vgg_16':               'vgg_16/vgg_16.ckpt',
                'vgg_19':               'vgg_19/vgg_19.ckpt',
                'inception_v1':         'inception_v1/inception_v1.ckpt',
                'inception_v2':         'inception_v2/inception_v2.ckpt',
                'inception_v3':         'inception_v3/inception_v3.ckpt',
                'inception_v4':         'inception_v4/inception_v4.ckpt',
                'resnet_v1_50':         'resnet_v1_50/resnet_v1_50.ckpt',
                'resnet_v1_101':        'resnet_v1_101/resnet_v1_101.ckpt',
                'resnet_v1_152':        'resnet_v1_152/resnet_v1_152.ckpt',
                'resnet_v2_50':         'resnet_v2_50/resnet_v2_50.ckpt',
                'resnet_v2_101':        'resnet_v2_101/resnet_v2_101.ckpt',
                'resnet_v2_152':        'resnet_v2_152/resnet_v2_152.ckpt',
                'mobilenet_v1':         'mobilenet_v1_1.0_128/',
                'mobilenet_v1_075':     'mobilenet_v1_0.75_128/',
                'mobilenet_v1_050':     'mobilenet_v1_0.50_128/',
                'mobilenet_v1_025':     'mobilenet_v1_0.25_128/',
               }

tf.app.flags.DEFINE_float(
    'weight_decay', 0.002, 'The weight decay on the model weights.')

FLAGS = flags.FLAGS
