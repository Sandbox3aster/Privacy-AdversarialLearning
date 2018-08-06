from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '..')

import errno
import itertools
import pprint
import time

import numpy as np
import os
import tensorflow as tf
from six.moves import xrange

from input_data import *
from nets import nets_factory

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

from slim_flags import FLAGS
from sklearn.metrics import average_precision_score
from vispr.utils import load_attributes, labels_to_vec
from degradNet import residualNet
import re

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, None, None, FLAGS.nchannel))
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.num_classes_budget))
    istraining_placeholder = tf.placeholder(tf.bool)
    return images_placeholder, labels_placeholder, istraining_placeholder

def tower_loss_xentropy_sparse(name_scope, logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cross_entropy_mean

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def accuracy(logits, labels):
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

def run_training(degrad_ckpt_file, ckpt_dir, model_name, max_steps, train_from_scratch, ckpt_path):
    batch_size = 128
    # Create model directory
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    continue_from_trained_model = False

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    network_fn = nets_factory.get_network_fn(model_name,
                                             num_classes=FLAGS.num_classes_budget,
                                             weight_decay=FLAGS.weight_decay,
                                             is_training=True)
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            images_placeholder, labels_placeholder, isTraining_placeholder = placeholder_inputs(batch_size * FLAGS.gpu_num)
            tower_grads = []
            logits_lst = []
            losses_lst = []
            opt = tf.train.AdamOptimizer(1e-4)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            degrad_images = residualNet(images_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size], training=False)

                            logits, _ = network_fn(degrad_images)
                            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                      labels=labels_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size, :]))
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

            grads = average_gradients(tower_grads)

            with tf.device('/cpu:%d' % 0):
                tvs = varlist_budget
                accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
                zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]


            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies([tf.group(*update_ops)]):
                accum_ops = [accum_vars[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads)]

            apply_gradient_op = opt.apply_gradients([(accum_vars[i].value(), gv[1]) for i, gv in enumerate(grads)], global_step=global_step)


            train_image_files = [os.path.join(FLAGS.train_images_files_dir, f) for
                           f in os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
            val_image_files = [os.path.join(FLAGS.val_images_files_dir, f) for
                         f in os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
            print('#############################Reading from files###############################')
            print(train_image_files)
            print(val_image_files)

            train_images_op, train_labels_op = inputs_images(filenames=train_image_files,
                                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                                   num_epochs=None,
                                                                   num_threads=FLAGS.num_threads,
                                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                                   shuffle=False)
            val_images_op, val_labels_op = inputs_images(filenames=val_image_files,
                                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                                   num_epochs=None,
                                                                   num_threads=FLAGS.num_threads,
                                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                                   shuffle=False)

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
                loss_value_lst = []
                sess.run(zero_ops)
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    train_images, train_labels = sess.run(
                        [train_images_op, train_labels_op])
                    _, loss_value = sess.run([accum_ops, loss_op], feed_dict={images_placeholder: train_images,
                                                                              labels_placeholder: train_labels,
                                                                              isTraining_placeholder: True})
                    loss_value_lst.append(loss_value)
                sess.run(apply_gradient_op)
                assert not np.isnan(np.mean(loss_value_lst)), 'Model diverged with loss = NaN'
                duration = time.time() - start_time
                print('Step: {:4d} time: {:.4f} loss: {:.8f}'.format(step, duration, np.mean(loss_value_lst)))
                if step % FLAGS.val_step == 0:
                    loss_budget_lst = []
                    pred_probs_lst = []
                    gt_lst = []
                    for _ in itertools.repeat(None, 30):
                        val_images, val_labels = sess.run([val_images_op, val_labels_op])
                        gt_lst.append(val_labels)
                        logits_budget, loss_budget = sess.run([logits_op, loss_op],
                                                              feed_dict={images_placeholder: val_images,
                                                                         labels_placeholder: val_labels,
                                                                         isTraining_placeholder: False})
                        loss_budget_lst.append(loss_budget)
                        pred_probs_lst.append(logits_budget)

                    pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
                    gt_mat = np.concatenate(gt_lst, axis=0)
                    n_examples, n_labels = gt_mat.shape
                    print('# Examples = ', n_examples)
                    print('# Labels = ', n_labels)
                    print('Macro MAP = {:.2f}'.format(
                        100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))

                # Save a checkpoint and evaluate the model periodically.
                if step % FLAGS.save_step == 0 or (step + 1) == max_steps:
                    checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

    print("done")

def run_testing(degrad_ckpt_file, ckpt_dir, model_name, is_training):
    batch_size = 128
    # Create model directory
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=FLAGS.num_classes_budget,
        weight_decay=FLAGS.weight_decay,
        is_training=True)
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            images_placeholder, labels_placeholder, isTraining_placeholder = placeholder_inputs(batch_size * FLAGS.gpu_num)
            logits_lst = []
            losses_lst = []
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            degrad_images = residualNet(images_placeholder[gpu_index * FLAGS.image_batch_size:(gpu_index + 1) * FLAGS.image_batch_size], training=False)
                            logits, _ = network_fn(degrad_images)
                            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                      labels=labels_placeholder[gpu_index * batch_size:(gpu_index + 1) * batch_size, :]))
                            logits_lst.append(logits)
                            losses_lst.append(loss)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
            loss_op = tf.reduce_mean(losses_lst)
            logits_op = tf.concat(logits_lst, 0)

            train_image_files = [os.path.join(FLAGS.train_images_files_dir, f) for
                                 f in os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
            test_image_files = [os.path.join(FLAGS.test_images_files_dir, f) for
                           f in os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
            print('#############################Reading from files###############################')
            print(test_image_files)

            if is_training:
                images_op, labels_op = inputs_images(filenames=train_image_files,
                                                     batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                     num_epochs=1,
                                                     num_threads=FLAGS.num_threads,
                                                     num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                     shuffle=False)
            else:
                images_op, labels_op = inputs_images(filenames=test_image_files,
                                                                   batch_size=FLAGS.image_batch_size * FLAGS.gpu_num,
                                                                   num_epochs=1,
                                                                   num_threads=FLAGS.num_threads,
                                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch,
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
            saver = tf.train.Saver(varlist_budget + bn_moving_vars)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from pretrained budget model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ckpt_dir)

            loss_budget_lst = []
            pred_probs_lst = []
            gt_lst = []
            try:
                while not coord.should_stop():
                    images, labels = sess.run([images_op, labels_op])
                    gt_lst.append(labels)
                    feed = {images_placeholder: images, labels_placeholder: labels,
                            isTraining_placeholder: False}
                    logits, loss = sess.run([logits_op, loss_op], feed_dict=feed)
                    loss_budget_lst.append(loss)
                    pred_probs_lst.append(logits)
            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()

            pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
            gt_mat = np.concatenate(gt_lst, axis=0)
            n_examples, n_labels = gt_mat.shape
            isTraining = lambda bool: "training" if bool else "validation"
            with open(os.path.join(ckpt_dir, '{}_{}_class_scores.txt'.format(model_name, isTraining(is_training))), 'w') as wf:
                wf.write('# Examples = {}\n'.format(n_examples))
                wf.write('# Labels = {}\n'.format(n_labels))
                wf.write('Macro MAP = {:.2f}\n'.format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))
                cmap_stats = average_precision_score(gt_mat, pred_probs_mat, average=None)
                attr_id_to_name, attr_id_to_idx = load_attributes()
                idx_to_attr_id = {v: k for k, v in attr_id_to_idx.items()}
                wf.write('\t'.join(['attribute_id', 'attribute_name', 'num_occurrences', 'ap']) + '\n')
                for idx in range(n_labels):
                    attr_id = idx_to_attr_id[idx]
                    attr_name = attr_id_to_name[attr_id]
                    attr_occurrences = np.sum(gt_mat, axis=0)[idx]
                    ap = cmap_stats[idx]
                    wf.write('{}\t{}\t{}\t{}\n'.format(attr_id, attr_name, attr_occurrences, ap * 100.0))

            coord.join(threads)
            sess.close()

    print("done")

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
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

    dir_path = FLAGS.ckpt_dir
    ckpt_files = [".".join(f.split(".")[:-1]) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and '.data' in f]
    for ckpt_file in ckpt_files:
        for model_name in model_name_lst:
            eval_ckpt_dir = 'checkpoint_eval/{}/{}/{}'.format(dir_path.split('/')[-1], ckpt_file.split('.')[-1], model_name)
            if not os.path.exists(eval_ckpt_dir):
                os.makedirs(eval_ckpt_dir)
            run_training(degrad_ckpt_file = os.path.join(dir_path, ckpt_file), ckpt_dir = eval_ckpt_dir, model_name = model_name, max_steps = model_max_steps_map[model_name],
                     train_from_scratch = model_train_from_scratch_map[model_name], ckpt_path = ckpt_path_map[model_name])
            run_testing(degrad_ckpt_file = os.path.join(dir_path, ckpt_file), ckpt_dir = eval_ckpt_dir, model_name = model_name, is_training=True)
            run_testing(degrad_ckpt_file = os.path.join(dir_path, ckpt_file), ckpt_dir = eval_ckpt_dir, model_name = model_name, is_training=False)

if __name__ == '__main__':
  tf.app.run()
