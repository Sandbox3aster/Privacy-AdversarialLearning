'''
Two-Fold-Evaluation
First-fold: action (utility) prediction performance is preserved
Second-fold: privacy (budget) prediction performance is suppressed
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '..')

import errno
import pprint
import time
import yaml

import numpy as np
import os
import tensorflow as tf
from six.moves import xrange

from input_data import *
from nets import nets_factory

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import sys
sys.path.insert(0, '..')

from degradlNet import residualNet
from loss import *
import re

def placeholder_inputs(batch_size, cfg):
    videos_placeholder = tf.placeholder(tf.float32, shape=(batch_size, cfg['DATA']['DEPTH'], None, None, cfg['DATA']['NCHANNEL']))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    istraining_placeholder = tf.placeholder(tf.bool)
    return videos_placeholder, labels_placeholder, istraining_placeholder

def accuracy(logits, labels):
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def run_training(cfg, degrad_ckpt_file, ckpt_dir, model_name, max_steps, train_from_scratch, ckpt_path):
    # Create model directory
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    continue_from_trained_model = False

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    network_fn = nets_factory.get_network_fn(model_name,
                                             num_classes=cfg['DATA']['NUM_CLASSES'],
                                             weight_decay=cfg['TRAIN']['WEIGHT_DECAY'],
                                             is_training=True)
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder, labels_placeholder, istraining_placeholder = placeholder_inputs(cfg['DATA']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'])
            tower_grads = []
            logits_lst = []
            losses_lst = []
            opt = tf.train.AdamOptimizer(1e-4)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            degrad_videos = residualNet(videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], is_video=True)
                            degrad_videos = tf.reshape(degrad_videos, [cfg['TRAIN']['BATCH_SIZE'] * cfg['DATA']['DEPTH'], cfg['DATA']['HEIGHT'], cfg['DATA']['WIDTH'], cfg['DATA']['NCHANNEL']])

                            logits, _ = network_fn(degrad_videos)
                            logits = tf.reshape(logits, [-1, cfg['DATA']['DEPTH'], cfg['DATA']['NUM_CLASSES']])
                            logits = tf.reduce_mean(logits, axis=1, keep_dims=False)
                            logits_lst.append(logits)
                            loss = tower_loss_xentropy_sparse(scope, logits,
                                    labels=labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE'], :])

                            logits_lst.append(logits)
                            losses_lst.append(loss)
                            print([v.name for v in tf.trainable_variables()])
                            varlist_budget = [v for v in tf.trainable_variables() if
                                              any(x in v.name for x in ["InceptionV1", "InceptionV2",
                                              "resnet_v1_50", "resnet_v1_101", "resnet_v2_50", "resnet_v2_101",
                                              "MobilenetV1_1.0", "MobilenetV1_0.75", "MobilenetV1_0.5", 'MobilenetV1_0.25'])]

                            varlist_degrad = [v for v in tf.trainable_variables() if v not in varlist_budget]
                            tower_grads.append(opt.compute_gradients(loss, varlist_budget))
                            tf.get_variable_scope().reuse_variables()
            loss_op = tf.reduce_mean(losses_lst)
            logits_op = tf.concat(logits_lst, 0)
            acc_op = accuracy(logits_op, labels_placeholder)
            grads = average_gradients(tower_grads)



            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies([tf.group(*update_ops)]):
                train_op = opt.apply_gradients(grads, global_step=global_step)


            train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for
                           f in os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
            val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for
                         f in os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]
            print('#############################Reading from files###############################')
            print(train_files)
            print(val_files)

            tr_videos_op, _, tr_labels_op = inputs_videos(filenames = train_files,
                                                 batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['BATCH_SIZE'],
                                                 num_epochs=None,
                                                 num_threads=cfg['DATA']['NUM_THREADS'],
                                                 num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                 shuffle=True)
            val_videos_op, _, val_labels_op = inputs_videos(filenames = val_files,
                                                 batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['BATCH_SIZE'],
                                                 num_epochs=None,
                                                 num_threads=cfg['DATA']['NUM_THREADS'],
                                                 num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                 shuffle=True)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            print([var.name for var in bn_moving_vars])


            def restore_model(dir, varlist, modulename):
                import re
                regex = re.compile(r'(MobilenetV1_?)(\d*\.?\d*)', re.IGNORECASE)
                if 'mobilenet' in modulename:
                    varlist = {regex.sub('MobilenetV1', v.name[:-2]): v for v in varlist}
                if os.path.isfile(dir):
                    print(varlist)
                    saver = tf.train.Saver(varlist)
                    saver.restore(sess, dir)
                    print('#############################Session restored from pretrained model at {}!#############################'.format(dir))
                else:
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver = tf.train.Saver(varlist)
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print('#############################Session restored from pretrained model at {}!#############################'.format(
                            ckpt.model_checkpoint_path))


            if continue_from_trained_model:
                varlist = varlist_budget
                varlist += bn_moving_vars
                saver = tf.train.Saver(varlist)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(
                        '#############################Session restored from trained model at {}!###############################'.format(
                            ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)
            else:
                if not train_from_scratch:
                    saver = tf.train.Saver(varlist_degrad)
                    print(degrad_ckpt_file)
                    saver.restore(sess, degrad_ckpt_file)


                    varlist = [v for v in varlist_budget+bn_moving_vars if not any(x in v.name for x in ["logits"])]
                    restore_model(ckpt_path, varlist, model_name)


            saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars, max_to_keep=1)
            for step in xrange(max_steps):
                start_time = time.time()
                train_videos, train_labels = sess.run([tr_videos_op, tr_labels_op])
                _, loss_value = sess.run([train_op, loss_op], feed_dict={videos_placeholder: train_videos,
                                                                              labels_placeholder: train_labels,
                                                                              istraining_placeholder: True})
                assert not np.isnan(np.mean(loss_value)), 'Model diverged with loss = NaN'
                duration = time.time() - start_time
                print('Step: {:4d} time: {:.4f} loss: {:.8f}'.format(step, duration, np.mean(loss_value)))
                if step % cfg['TRAIN']['VAL_STEP'] == 0:
                    start_time = time.time()
                    tr_videos, tr_labels = sess.run(
                        [tr_videos_op, tr_labels_op])
                    acc, loss_value = sess.run([acc_op, loss_op],
                                               feed_dict={videos_placeholder: tr_videos,
                                                          labels_placeholder: tr_labels,
                                                          istraining_placeholder: False})
                    print("Step: {:4d} time: {:.4f}, training accuracy: {:.5f}, loss: {:.8f}".
                          format(step, time.time() - start_time, acc, loss_value))

                    # train_writer.add_summary(summary, step)

                    start_time = time.time()
                    val_videos, val_labels = sess.run(
                        [val_videos_op, val_labels_op])
                    acc, loss_value = sess.run([acc_op, loss_op],
                                               feed_dict={videos_placeholder: val_videos,
                                                          labels_placeholder: val_labels,
                                                          istraining_placeholder: False})
                    print("Step: {:4d} time: {:.4f}, validation accuracy: {:.5f}, loss: {:.8f}".
                          format(step, time.time() - start_time, acc, loss_value))
                    # test_writer.add_summary(summary, step)

                    # Save a checkpoint and evaluate the model periodically.
                if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == cfg['TRAIN']['MAX_STEPS']:
                    checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

    print("done")

def run_testing(cfg, degrad_ckpt_file, ckpt_dir, model_name, is_training):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    network_fn = nets_factory.get_network_fn(model_name,
                                             num_classes=cfg['DATA']['NUM_CLASSES'],
                                             weight_decay=cfg['TRAIN']['WEIGHT_DECAY'],
                                             is_training=True)
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder, _, labels_placeholder, _, _ = placeholder_inputs(cfg['DATA']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'])
            istraining_placeholder = tf.placeholder(tf.bool)
            logits_lst = []
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            degrad_videos = residualNet(videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], is_video=True)
                            degrad_videos = tf.reshape(degrad_videos, [cfg['TRAIN']['BATCH_SIZE'] * cfg['DATA']['DEPTH'], cfg['DATA']['HEIGHT'], cfg['DATA']['WIDTH'], cfg['DATA']['NCHANNEL']])
                            logits, _ = network_fn(degrad_videos)
                            logits = tf.reshape(logits, [-1, cfg['DATA']['DEPTH'], cfg['DATA']['NUM_CLASSES']])
                            logits = tf.reduce_mean(logits, axis=1, keep_dims=False)
                            logits_lst.append(logits)
                            tf.get_variable_scope().reuse_variables()
            logits_op = tf.concat(logits_lst, 0)

            right_count_op = tf.reduce_sum(
                tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_op), axis=1), labels_placeholder), tf.int32))
            softmax_logits_op = tf.nn.softmax(logits_op)

            train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for
                           f in os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
            val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for
                         f in os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]
            print('#############################Reading from files###############################')
            print(train_files)
            print(val_files)
            print('#############################Reading from files###############################')
            print(train_files)
            print(val_files)

            if is_training:
                videos_op, _, labels_op = inputs_videos(filenames = train_files,
                                                 batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['BATCH_SIZE'],
                                                 num_epochs=None,
                                                 num_threads=cfg['DATA']['NUM_THREADS'],
                                                 num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                 shuffle=False)
            else:
                videos_op, _, labels_op = inputs_videos(filenames = val_files,
                                                 batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['BATCH_SIZE'],
                                                 num_epochs=None,
                                                 num_threads=cfg['DATA']['NUM_THREADS'],
                                                 num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                 shuffle=False)


            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            varlist_budget = [v for v in tf.trainable_variables() if
                              any(x in v.name for x in ["InceptionV1", "InceptionV2",
                                                        "resnet_v1_50", "resnet_v1_101", "resnet_v2_50",
                                                        "resnet_v2_101",
                                                        "MobilenetV1_1.0", "MobilenetV1_0.75", "MobilenetV1_0.5",
                                                        'MobilenetV1_0.25'])]

            varlist_degrad = [v for v in tf.trainable_variables() if v not in varlist_budget]

            saver = tf.train.Saver(varlist_degrad)
            saver.restore(sess, degrad_ckpt_file)

            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from pretrained budget model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)
            total_v = 0.0
            test_correct_num = 0.0
            try:
                while not coord.should_stop():
                    videos, labels = sess.run([videos_op, labels_op])
                    # write_video(videos, labels)
                    feed = {videos_placeholder: videos, labels_placeholder: labels,
                            istraining_placeholder: False}
                    right, softmax_logits = sess.run([right_count_op, softmax_logits_op], feed_dict=feed)
                    test_correct_num += right
                    total_v += labels.shape[0]
                    print(softmax_logits.shape)
                    # print(tf.argmax(softmax_logits, 1).eval(session=sess))
                    # print(logits.eval(feed_dict=feed, session=sess))
                    # print(labels)
            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()
            print('test acc:', test_correct_num / total_v, 'test_correct_num:', test_correct_num,
                  'total_v:', total_v)
            with open('TwoFoldEvaluationResults_{}.txt'.format(model_name), 'w') as wf:
                wf.write('test acc: {}\ttest_correct_num:{}\ttotal_v\n'.format(
                    test_correct_num / total_v, test_correct_num, total_v))
            coord.join(threads)
            sess.close()

    print("done")

def main(_):
    cfg = yaml.load(open('params.yml'))
    pp = pprint.PrettyPrinter()
    pp.pprint(cfg)

    ckpt_base = '../evaluation_models/{}'

    ckpt_path_map = {
        'inception_v1': ckpt_base.format('inception_v1/inception_v1.ckpt'),
        'inception_v2': ckpt_base.format('inception_v2/inception_v2.ckpt'),
        'resnet_v1_50': ckpt_base.format('resnet_v1_50/resnet_v1_50.ckpt'),
        'resnet_v1_101': ckpt_base.format('resnet_v1_101/resnet_v1_101.ckpt'),
        'resnet_v2_50': ckpt_base.format('resnet_v2_50/resnet_v2_50.ckpt'),
        'resnet_v2_101': ckpt_base.format('resnet_v2_101/resnet_v2_101.ckpt'),
        'mobilenet_v1': ckpt_base.format('mobilenet_v1_1.0_128/'),
        'mobilenet_v1_075': ckpt_base.format('mobilenet_v1_0.75_128/'),
        'mobilenet_v1_050': ckpt_base.format('mobilenet_v1_0.50_128/'),
        'mobilenet_v1_025': ckpt_base.format('mobilenet_v1_0.25_128/'),
    }
    model_max_steps_map = {
        'inception_v1': 400,
        'inception_v2': 400,
        'resnet_v1_50': 400,
        'resnet_v1_101': 400,
        'resnet_v2_50': 400,
        'resnet_v2_101': 400,
        'mobilenet_v1': 400,
        'mobilenet_v1_075': 400,
        'mobilenet_v1_050': 1000,
        'mobilenet_v1_025': 1000,
    }
    # Whether we need to train from scratch
    # Among the 10 evaluation models, 8 starts from imagenet pretrained model and 2 starts from scratch
    model_train_from_scratch_map = {
        'inception_v1': False,
        'inception_v2': False,
        'resnet_v1_50': False,
        'resnet_v1_101': False,
        'resnet_v2_50': False,
        'resnet_v2_101': False,
        'mobilenet_v1': False,
        'mobilenet_v1_075': False,
        'mobilenet_v1_050': True,
        'mobilenet_v1_025': True,
    }
    model_name_lst = ['mobilenet_v1', 'mobilenet_v1_075', 'mobilenet_v1_050', 'mobilenet_v1_025',
                      'resnet_v1_50', 'resnet_v1_101', 'resnet_v2_50', 'resnet_v2_101',
                      'inception_v1', 'inception_v2']

    dir_path = cfg['MODEL']['CKPT_DIR']
    # Evaluation all the models in the specified directory
    ckpt_files = [".".join(f.split(".")[:-1]) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and '.data' in f]
    for ckpt_file in ckpt_files:
        for model_name in model_name_lst:
            eval_ckpt_dir = 'checkpoint_eval/{}/{}/{}'.format(dir_path.split('/')[-1], ckpt_file.split('.')[-1], model_name)
            if not os.path.exists(eval_ckpt_dir):
                os.makedirs(eval_ckpt_dir)
            run_training(cfg, degrad_ckpt_file = os.path.join(dir_path, ckpt_file), ckpt_dir = eval_ckpt_dir, model_name = model_name, max_steps = model_max_steps_map[model_name],
                     train_from_scratch = model_train_from_scratch_map[model_name], ckpt_path = ckpt_path_map[model_name])
            run_testing(cfg, degrad_ckpt_file = os.path.join(dir_path, ckpt_file), ckpt_dir = eval_ckpt_dir, model_name = model_name, is_training=True)
            run_testing(cfg, degrad_ckpt_file = os.path.join(dir_path, ckpt_file), ckpt_dir = eval_ckpt_dir, model_name = model_name, is_training=False)

if __name__ == '__main__':
  tf.app.run()
