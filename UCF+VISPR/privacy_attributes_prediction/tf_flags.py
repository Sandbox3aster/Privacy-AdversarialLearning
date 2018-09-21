import tensorflow as tf

module_name = 'resnet_v2_101'


flags = tf.app.flags

flags.DEFINE_string('checkpoint_dir', '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/DAN/checkpoint/budget_multi_models/17_attributes', 'Directory where to read/write model checkpoints')

flags.DEFINE_integer('depth', 16, 'Depth for the video')
flags.DEFINE_integer('width', 160, 'Width for the video')
flags.DEFINE_integer('height', 120, 'Height for the video')
flags.DEFINE_integer('crop_width', 112, 'Width for the video')
flags.DEFINE_integer('crop_height', 112, 'Height for the video')
flags.DEFINE_boolean('use_random_crop', True, 'Whether to use random crop when reading video in the input pipeline')
flags.DEFINE_integer('nchannel', 3 ,'Number of channel for the video')
flags.DEFINE_integer('n_minibatches', 8, 'Number of mini-batches')

flags.DEFINE_string('train_images_files_dir', '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/vispr_tfrecords/17attributes/train', 'The directory of the training files')
flags.DEFINE_string('val_images_files_dir', '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/vispr_tfrecords/17attributes/val', 'The directory of the validation files')
flags.DEFINE_string('test_images_files_dir', '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/vispr_tfrecords/17attributes/test', 'The directory of the testing files')


flags.DEFINE_integer('gpu_num', 4, 'Number of gpus to run')

flags.DEFINE_integer('num_examples_per_epoch', 500, 'Epoch size')
flags.DEFINE_integer('num_threads', 10, 'Number of threads enqueuing tensor list')
log_dir = 'tensorboard_events/'
flags.DEFINE_string('log_dir', log_dir, 'Directory where to write the tensorboard events')
flags.DEFINE_integer('max_steps', 400, 'Number of steps to run trainer.')
flags.DEFINE_integer('val_step', 10, 'Number of steps for validation')
flags.DEFINE_integer('save_step', 25, 'Number of step to save the model')



#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.001, 'The weight decay on the model weights.')

flags.DEFINE_string(
    'model_name', module_name, 'The name of the architecture to train.')

flags.DEFINE_integer(
    'batch_size', 64, 'The number of samples in each batch.')

flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

flags.DEFINE_integer('num_classes', 17, 'Number of classes to do classification')

#####################
# Fine-Tuning Flags #
#####################

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

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/wuzhenyu_sjtu/DAN_vispr_ucf/evaluation_models/{}'.format(ckpt_path_map[module_name]),
    'The path to a checkpoint from which to fine-tune.')

FLAGS = flags.FLAGS