from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import itertools
import pprint
import time

import numpy as np
import os
import tensorflow as tf
from six.moves import xrange

import input_data
from nets import nets_factory
from vispr.utils import *
from sklearn.metrics import average_precision_score

slim = tf.contrib.slim
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

from tf_flags import FLAGS

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
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

def run_training():
    # Create model directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.gpu_num*FLAGS.batch_size, None, None, FLAGS.nchannel))
            labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.gpu_num*FLAGS.batch_size, FLAGS.num_classes))
            istraining_placeholder = tf.placeholder(tf.bool)
            tower_grads = []
            logits_lst = []
            losses_lst = []
            learning_rate = tf.train.exponential_decay(
                0.001,  # Base learning rate.
                global_step,  # Current index into the dataset.
                5000,  # Decay step.
                0.96,  # Decay rate.
                staircase=True)
            # Use simple momentum for the optimization.
            #opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
            opt = tf.train.AdamOptimizer(1e-3)
            network_fn = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=FLAGS.num_classes,
                weight_decay=FLAGS.weight_decay,
                is_training=istraining_placeholder)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            X = images_placeholder[gpu_index * FLAGS.batch_size:
                                                        (gpu_index + 1) * FLAGS.batch_size]
                            logits, _ = network_fn(X)
                            logits_lst.append(logits)
                            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,
                                                labels = labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]))
                            losses_lst.append(loss)
                            #varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["logits"])]
                            varlist = tf.trainable_variables()
                            print([v.name for v in varlist])
                            grads = opt.compute_gradients(loss, varlist)
                            tower_grads.append(grads)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
            loss_op = tf.reduce_mean(losses_lst, name='softmax')
            logits_op = tf.concat(logits_lst, 0)
            grads = average_gradients(tower_grads)

            with tf.device('/cpu:%d' % 0):
                tvs = varlist
                accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in
                                         tvs]
                zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies([tf.group(*update_ops)]):
                accum_ops = [accum_vars[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads)]


            apply_gradient_op = opt.apply_gradients([(accum_vars[i].value(), gv[1]) for i, gv in enumerate(grads)], global_step=global_step)

            train_files = [os.path.join(FLAGS.train_images_files_dir, f) for
                           f in os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
            val_files = [os.path.join(FLAGS.val_images_files_dir, f) for
                         f in os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
            test_files = [os.path.join(FLAGS.test_images_files_dir, f) for
                         f in os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
            print('#############################Reading from files###############################')
            print(train_files)
            print(val_files)

            tr_images_op, tr_labels_op = input_data.inputs(filenames = train_files,
                                                              batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                              num_epochs=None,
                                                              num_threads=FLAGS.num_threads,
                                                              num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                              shuffle=True,
                                                              distort=True,
                                                        )
            val_images_op, val_labels_op = input_data.inputs(filenames = val_files,
                                                                batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                num_epochs=None,
                                                                num_threads=FLAGS.num_threads,
                                                                num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                                shuffle=True,
                                                                distort=False,
                                                             )
            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            print([var.name for var in bn_moving_vars])
            # Create a saver for writing training checkpoints.

            if use_pretrained_model:
                varlist = [v for v in tf.trainable_variables() if not any(x in v.name for x in ["logits"])]
                #varlist += bn_moving_vars
                #vardict = {v.name[:-2].replace('MobileNet', 'MobilenetV1'): v for v in varlist}
                saver = tf.train.Saver(varlist)
                #saver = tf.train.Saver(vardict)
                if os.path.isfile(FLAGS.checkpoint_path):
                    saver.restore(sess, FLAGS.checkpoint_path)
                    print('#############################Session restored from pretrained model at {}!###############################'.format(FLAGS.checkpoint_path))
                else:
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_path)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver = tf.train.Saver(varlist)
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print('Session restored from pretrained degradation model at {}!'.format(
                            ckpt.model_checkpoint_path))
                    else:
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_path)
            else:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver = tf.train.Saver(tf.trainable_variables())
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from pretrained degradation model at {}!'.format(
                        ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)


            # Create summary writter
            saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                loss_value_lst = []
                sess.run(zero_ops)
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_labels = sess.run(
                        [tr_images_op, tr_labels_op])
                    _, loss_value = sess.run(
                        [accum_ops, loss_op],
                        feed_dict={images_placeholder: tr_videos,
                                   labels_placeholder: tr_labels,
                                   istraining_placeholder: True})
                    loss_value_lst.append(loss_value)
                sess.run(apply_gradient_op)
                assert not np.isnan(np.mean(loss_value_lst)), 'Model diverged with loss = NaN'
                duration = time.time() - start_time
                print('Step: {:4d} time: {:.4f} loss: {:.8f}'.format(step, duration, np.mean(loss_value_lst)))
                if step % FLAGS.val_step == 0:
                    start_time = time.time()
                    tr_videos, tr_labels= sess.run(
                            [tr_images_op, tr_labels_op])
                    loss_value = sess.run(loss_op, feed_dict={images_placeholder: tr_videos,
                                                                labels_placeholder: tr_labels,
                                                                istraining_placeholder: True})
                    print("Step: {:4d} time: {:.4f}, training loss: {:.8f}".format(step, time.time()-start_time,  loss_value))


                    start_time = time.time()
                    val_videos, val_labels = sess.run(
                            [val_images_op, val_labels_op])
                    loss_value = sess.run(loss_op, feed_dict={images_placeholder: val_videos,
                                                                labels_placeholder: val_labels,
                                                                istraining_placeholder: True})
                    print("Step: {:4d} time: {:.4f}, validation loss: {:.8f}".format(step, time.time() - start_time, loss_value))

                # Save a checkpoint and evaluate the model periodically.
                if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

    print("done")

def run_testing():
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.gpu_num*FLAGS.batch_size, None, None, FLAGS.nchannel))
            labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.gpu_num*FLAGS.batch_size, FLAGS.num_classes))
            istraining_placeholder = tf.placeholder(tf.bool)
            logits_lst = []
            network_fn = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=FLAGS.num_classes,
                weight_decay=FLAGS.weight_decay,
                is_training=istraining_placeholder)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            X = images_placeholder[gpu_index * FLAGS.batch_size:
                            (gpu_index + 1) * FLAGS.batch_size]
                            logits, _ = network_fn(X)
                            logits_lst.append(logits)
                            tf.get_variable_scope().reuse_variables()
            logits_op = tf.concat(logits_lst, 0)

            train_files = [os.path.join(FLAGS.train_images_files_dir, f) for
                           f in os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
            val_files = [os.path.join(FLAGS.val_images_files_dir, f) for
                         f in os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
            test_files = [os.path.join(FLAGS.test_images_files_dir, f) for
                         f in os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
            print('#############################Reading from files###############################')
            print(train_files)
            print(val_files)

            images_op, labels_op = input_data.inputs(filenames = test_files,
                                                              batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                              num_epochs=1,
                                                              num_threads=FLAGS.num_threads,
                                                              num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                              shuffle=True)

            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('----------------------------Trainable Variables-----------------------------------------')
            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from pretrained budget model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)
            pred_probs_lst = []
            gt_lst = []
            try:
                while not coord.should_stop():
                    images, labels = sess.run([images_op, labels_op])
                    # write_video(videos, labels)
                    gt_lst.append(labels)
                    feed = {images_placeholder: images, labels_placeholder: labels,
                            istraining_placeholder: True}

                    logits = sess.run(logits_op, feed_dict=feed)
                    pred_probs_lst.append(logits)
                    #print(logits)
                    # print(tf.argmax(softmax_logits, 1).eval(session=sess))
                    # print(logits.eval(feed_dict=feed, session=sess))
                    # print(labels)
            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()

            pred_probs_mat = np.concatenate(pred_probs_lst, axis=0)
            gt_mat = np.concatenate(gt_lst, axis=0)
            n_examples, n_labels = gt_mat.shape
            print('# Examples = ', n_examples)
            print('# Labels = ', n_labels)
            print('Macro MAP = {:.2f}'.format(100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))
            cmap_stats = average_precision_score(gt_mat, pred_probs_mat, average=None)
            attr_id_to_name, attr_id_to_idx = load_attributes()
            idx_to_attr_id = {v: k for k, v in attr_id_to_idx.items()}
            with open('class_scores.txt', 'w') as wf:
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

def run_training_multi_models():
    # Create model directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.gpu_num*FLAGS.batch_size, None, None, FLAGS.nchannel))
            labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.gpu_num*FLAGS.batch_size, FLAGS.num_classes))
            istraining_placeholder = tf.placeholder(tf.bool)
            tower_grads = []
            logits_lst = []
            losses_lst = []
            # learning_rate = tf.train.exponential_decay(
            #     0.001,  # Base learning rate.
            #     global_step,  # Current index into the dataset.
            #     5000,  # Decay step.
            #     0.96,  # Decay rate.
            #     staircase=True)
            # # Use simple momentum for the optimization.
            #opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
            opt = tf.train.AdamOptimizer(1e-4)
            model_dict = {}
            model_name_lst = ['resnet_v1_50','resnet_v2_50','mobilenet_v1','mobilenet_v1_075']
            for model_name in model_name_lst:
                model_dict[model_name] = nets_factory.get_network_fn(
                                                                model_name,
                                                                num_classes=FLAGS.num_classes,
                                                                weight_decay=FLAGS.weight_decay,
                                                                is_training=True)

            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            X = images_placeholder[gpu_index * FLAGS.batch_size:
                                                        (gpu_index + 1) * FLAGS.batch_size]
                            loss_budget = 0.0
                            logits_budget = tf.zeros([FLAGS.batch_size, FLAGS.num_classes])

                            for model_name in model_name_lst:
                                print(model_name)
                                logits, _ = model_dict[model_name](X)
                                logits_budget += logits
                                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,
                                                labels = labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]))
                                loss_budget += loss

                            logits_lst.append(logits_budget)
                            losses_lst.append(loss_budget)
                            #varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["logits"])]
                            varlist = tf.trainable_variables()
                            print([v.name for v in varlist])
                            grads = opt.compute_gradients(loss_budget, varlist)
                            tower_grads.append(grads)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
            loss_op = tf.reduce_mean(losses_lst, name='softmax')
            logits_op = tf.concat(logits_lst, 0)
            grads = average_gradients(tower_grads)

            with tf.device('/cpu:%d' % 0):
                tvs = varlist
                accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in
                                         tvs]
                zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            with tf.control_dependencies([tf.group(*update_ops)]):
                accum_ops = [accum_vars[i].assign_add(gv[0]/FLAGS.n_minibatches) for i, gv in enumerate(grads)]


            apply_gradient_op = opt.apply_gradients([(accum_vars[i].value(), gv[1]) for i, gv in enumerate(grads)], global_step=global_step)

            train_files = [os.path.join(FLAGS.train_images_files_dir, f) for
                           f in os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
            val_files = [os.path.join(FLAGS.val_images_files_dir, f) for
                         f in os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
            test_files = [os.path.join(FLAGS.test_images_files_dir, f) for
                         f in os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
            print('#############################Reading from files###############################')
            print(train_files)
            print(val_files)

            tr_images_op, tr_labels_op = input_data.inputs(filenames = train_files,
                                                              batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                              num_epochs=None,
                                                              num_threads=FLAGS.num_threads,
                                                              num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                              shuffle=True,
                                                              distort=True,
                                                        )
            val_images_op, val_labels_op = input_data.inputs(filenames = val_files,
                                                                batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                                num_epochs=None,
                                                                num_threads=FLAGS.num_threads,
                                                                num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                                shuffle=True,
                                                                distort=False,
                                                             )
            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            print([var.name for var in bn_moving_vars])
            # Create a saver for writing training checkpoints.
            ckpt_path_map = {
                'vgg_16': 'vgg_16/vgg_16.ckpt',
                'vgg_19': 'vgg_19/vgg_19.ckpt',
                'inception_v1': 'inception_v1/inception_v1.ckpt',
                'inception_v2': 'inception_v2/inception_v2.ckpt',
                'inception_v3': 'inception_v3/inception_v3.ckpt',
                'inception_v4': 'inception_v4/inception_v4.ckpt',
                'resnet_v1_50': 'resnet_v1_50/resnet_v1_50.ckpt',
                'resnet_v1_101': 'resnet_v1_101/resnet_v1_101.ckpt',
                'resnet_v1_152': 'resnet_v1_152/resnet_v1_152.ckpt',
                'resnet_v2_50': 'resnet_v2_50/resnet_v2_50.ckpt',
                'resnet_v2_101': 'resnet_v2_101/resnet_v2_101.ckpt',
                'resnet_v2_152': 'resnet_v2_152/resnet_v2_152.ckpt',
                'mobilenet_v1': 'mobilenet_v1_1.0_128/',
                'mobilenet_v1_075': 'mobilenet_v1_0.75_128/',
                'mobilenet_v1_050': 'mobilenet_v1_0.50_128/',
                'mobilenet_v1_025': 'mobilenet_v1_0.25_128/',
            }


            def restore_model(dir, varlist, modulename):
                import re
                regex = re.compile(r'(MobilenetV1_?)(\d*\.?\d*)', re.IGNORECASE)
                if 'mobilenet' in modulename:
                    varlist = {regex.sub('MobilenetV1', v.name[:-2]): v for v in varlist}
                saver = tf.train.Saver(varlist, max_to_keep=20)
                if os.path.isfile(dir):
                    print(varlist)
                    saver.restore(sess, dir)
                    print('#############################Session restored from pretrained model at {}!#############################'.format(dir))
                else:
                    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print('#############################Session restored from pretrained model at {}!#############################'.format(
                            ckpt.model_checkpoint_path))
                    else:
                        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dir)

            if use_pretrained_model:
                varlist_dict = {}
                eval_dir = '/home/wuzhenyu_sjtu/DAN_VISPR_UCF/evaluation_models/{}'
                model_name_mapping = {'resnet_v1_50':'resnet_v1_50', 'resnet_v2_50':'resnet_v2_50',
                                      'mobilenet_v1':'MobilenetV1_1.0','mobilenet_v1_075':'MobilenetV1_0.75'}
                for model_name in model_name_lst:
                    print(model_name)
                    varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in [model_name_mapping[model_name]])]
                    varlist_dict[model_name] = [v for v in varlist if not any(x in v.name for x in ["logits"])]
                    #print(varlist_dict[model_name])
                    print(ckpt_path_map[model_name])
                    restore_model(eval_dir.format(ckpt_path_map[model_name]), varlist_dict[model_name], model_name)
            else:
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver = tf.train.Saver(tf.trainable_variables())
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from pretrained degradation model at {}!'.format(
                        ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            # Create summary writter
            saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                loss_value_lst = []
                sess.run(zero_ops)
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_labels = sess.run(
                        [tr_images_op, tr_labels_op])
                    _, loss_value = sess.run([accum_ops, loss_op], feed_dict={images_placeholder: tr_videos,
                                                                              labels_placeholder: tr_labels,
                                                                              istraining_placeholder: True})
                    loss_value_lst.append(loss_value)
                sess.run(apply_gradient_op)
                assert not np.isnan(np.mean(loss_value_lst)), 'Model diverged with loss = NaN'
                duration = time.time() - start_time
                print('Step: {:4d} time: {:.4f} loss: {:.8f}'.format(step, duration, np.mean(loss_value_lst)))
                if step % FLAGS.val_step == 0:
                    start_time = time.time()
                    tr_videos, tr_labels= sess.run(
                            [tr_images_op, tr_labels_op])
                    loss_value = sess.run(loss_op, feed_dict={images_placeholder: tr_videos,
                                                                labels_placeholder: tr_labels,
                                                                istraining_placeholder: True})
                    print("Step: {:4d} time: {:.4f}, training loss: {:.8f}".format(step, time.time()-start_time,  loss_value))


                    start_time = time.time()
                    val_videos, val_labels = sess.run(
                            [val_images_op, val_labels_op])
                    loss_value = sess.run(loss_op, feed_dict={images_placeholder: val_videos,
                                                                labels_placeholder: val_labels,
                                                                istraining_placeholder: True})
                    print("Step: {:4d} time: {:.4f}, validation loss: {:.8f}".format(step, time.time() - start_time, loss_value))

                # Save a checkpoint and evaluate the model periodically.
                if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

    print("done")

def run_testing_multi_models(is_training=False):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.gpu_num*FLAGS.batch_size, None, None, FLAGS.nchannel))
            labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.gpu_num*FLAGS.batch_size, FLAGS.num_classes))
            isTraining_placeholder = tf.placeholder(tf.bool)

            from collections import defaultdict
            logits_budget_images_lst_dct = defaultdict(list)
            loss_budget_images_lst_dct = defaultdict(list)

            logits_lst = []
            losses_lst = []
            model_dict = {}
            model_name_lst = ['resnet_v1_50', 'resnet_v2_50', 'mobilenet_v1', 'mobilenet_v1_075']
            for model_name in model_name_lst:
                model_dict[model_name] = nets_factory.get_network_fn(
                    model_name,
                    num_classes=FLAGS.num_classes,
                    weight_decay=FLAGS.weight_decay,
                    is_training=True)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            X = images_placeholder[gpu_index * FLAGS.batch_size : (gpu_index + 1) * FLAGS.batch_size]
                            loss_budget = 0.0
                            logits_budget = tf.zeros([FLAGS.batch_size, FLAGS.num_classes])

                            for model_name in model_name_lst:
                                print(model_name)
                                print(tf.trainable_variables())
                                logits, _ = model_dict[model_name](X)
                                logits_budget += logits
                                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                                              labels=labels_placeholder[
                                                                                                     gpu_index * FLAGS.batch_size:(
                                                                                                                                  gpu_index + 1) * FLAGS.batch_size]))
                                loss_budget += loss
                                logits_budget_images_lst_dct[model_name].append(logits)
                                loss_budget_images_lst_dct[model_name].append(loss)
                            logits_budget = tf.divide(logits_budget, 4.0, 'LogitsBudgetMean')
                            logits_lst.append(logits_budget)
                            losses_lst.append(loss_budget)
                            # varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["logits"])]
                            varlist = tf.trainable_variables()
                            print([v.name for v in varlist])
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
            loss_op = tf.reduce_mean(losses_lst)
            logits_op = tf.concat(logits_lst, 0)

            logits_op_lst = []
            for model_name in model_name_lst:
                logits_op_lst.append(tf.concat(logits_budget_images_lst_dct[model_name], axis=0))

            train_files = [os.path.join(FLAGS.train_images_files_dir, f) for
                           f in os.listdir(FLAGS.train_images_files_dir) if f.endswith('.tfrecords')]
            val_files = [os.path.join(FLAGS.val_images_files_dir, f) for
                         f in os.listdir(FLAGS.val_images_files_dir) if f.endswith('.tfrecords')]
            test_files = [os.path.join(FLAGS.test_images_files_dir, f) for
                         f in os.listdir(FLAGS.test_images_files_dir) if f.endswith('.tfrecords')]
            print('#############################Reading from files###############################')
            print(train_files)
            print(val_files)
            print(test_files)

            if is_training:
                images_op, labels_op = input_data.inputs(filenames=train_files,
                                                         batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                         num_epochs=1,
                                                         num_threads=FLAGS.num_threads,
                                                         num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                         shuffle=False)
            else:
                images_op, labels_op = input_data.inputs(filenames = test_files,
                                                     batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                     num_epochs=1,
                                                     num_threads=FLAGS.num_threads,
                                                     num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                                     shuffle=False)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            print('----------------------------Trainable Variables-----------------------------------------')
            gvar_list = tf.global_variables()
            bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
            print(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from pretrained budget model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            model_name_lst += ['ensemble']
            loss_budget_lst = []
            pred_probs_lst_lst = [[] for _ in xrange(len(model_name_lst))]
            gt_lst = []
            try:
                while not coord.should_stop():
                    images, labels = sess.run([images_op, labels_op])
                    # write_video(videos, labels)
                    gt_lst.append(labels)
                    value_lst = sess.run([loss_op, logits_op] + logits_op_lst,
                                         feed_dict={images_placeholder: images,
                                                    labels_placeholder: labels,
                                                    isTraining_placeholder: True})
                    print(labels.shape)
                    loss_budget_lst.append(value_lst[0])
                    for i in xrange(len(model_name_lst)):
                        pred_probs_lst_lst[i].append(value_lst[i + 1])

            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()

            gt_mat = np.concatenate(gt_lst, axis=0)
            n_examples, n_labels = gt_mat.shape
            for i in xrange(len(model_name_lst)):
                save_dir = os.path.join(FLAGS.checkpoint_dir, 'evaluation')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                isTraining = lambda bool: "training" if bool else "validation"
                with open(os.path.join(save_dir, '{}_class_scores_{}.txt'.format(model_name_lst[i],
                                                                                 isTraining(is_training))), 'w') as wf:
                    pred_probs_mat = np.concatenate(pred_probs_lst_lst[i], axis=0)
                    wf.write('# Examples = {}\n'.format(n_examples))
                    wf.write('# Labels = {}\n'.format(n_labels))
                    wf.write('Average Loss = {}\n'.format(np.mean(loss_budget_lst)))
                    wf.write("Macro MAP = {:.2f}\n".format(
                        100 * average_precision_score(gt_mat, pred_probs_mat, average='macro')))
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
    #run_training()
    #run_training_multi_models()
    run_testing_multi_models(is_training=True)
    run_testing_multi_models(is_training=False)
    #run_testing()

if __name__ == '__main__':
  tf.app.run()
