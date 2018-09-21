#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="4,6"

import sys
sys.path.insert(0, '..')

import time
from six.moves import xrange
import input_data
import errno
import pprint
import itertools
from degradlNet import residualNet
from budgetNet import budgetNet
from utilityNet import utilityNet
from loss import *
from utils import *
from img_proc import _bilinear_resize, _avg_replicate, _binary_activation, _instance_norm
from visualization.visualize import plot_visualization_frame
import yaml
from tf_flags import FLAGS

def placeholder_inputs(cfg, batch_size):
    videos_placeholder = tf.placeholder(tf.float32, shape=(batch_size, cfg['DATA']['DEPTH'], 112, 112, cfg['DATA']['NCHANNEL']))
    action_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    actor_labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    dropout_placeholder = tf.placeholder(tf.float32)
    isTraining_placeholder = tf.placeholder(tf.bool)
    return videos_placeholder, action_labels_placeholder, actor_labels_placeholder, dropout_placeholder, isTraining_placeholder

# ResidualNet initialization (identity mapping)
def run_training_degradation_residual(cfg):
    if not os.path.exists(FLAGS.degradation_models):
        os.makedirs(FLAGS.degradation_models)
    start_from_trained_model = True

    with tf.Graph().as_default():
        global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
        images_placeholder = tf.placeholder(tf.float32, [cfg['TRAIN']['BATCH_SIZE']*cfg['TRAIN']['GPU_NUM'], 120, 160, 3], name='images')
        labels_placeholder = tf.placeholder(tf.float32, [cfg['TRAIN']['BATCH_SIZE']*cfg['TRAIN']['GPU_NUM'], 120, 160, 3], name='labels')
        tower_grads = []
        losses = []
        opt = tf.train.AdamOptimizer(1e-3)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                        pred = residualNet(images_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], is_video=True)
                        loss = tower_loss_mse(scope, pred,
                                          labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']])
                        losses.append(loss)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
                        tf.get_variable_scope().reuse_variables()
        loss_op = tf.reduce_mean(losses, name='mse')
        psnr_op = tf.multiply(tf.constant(20, dtype=tf.float32), tf.log(1 /tf.sqrt(loss_op))/tf.log(tf.constant(10, dtype=tf.float32)), name='psnr')

        tf.summary.scalar('loss', loss_op)

        grads_avg = average_gradients(tower_grads)
        train_op = opt.apply_gradients(grads_avg)


        train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DEG_DIR'], f) for f in
                       os.listdir(cfg['DATA']['TRAIN_FILES_DEG_DIR']) if f.endswith('.tfrecords')]
        val_files = [os.path.join(cfg['DATA']['VAL_FILES_DEG_DIR'], f) for f in
                     os.listdir(cfg['DATA']['VAL_FILES_DEG_DIR']) if f.endswith('.tfrecords')]

        tr_images_op, tr_labels_op = input_data.inputs_images(filenames = train_files,
                                                 batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                 num_epochs=None,
                                                 num_threads=cfg['DATA']['NUM_THREADS'],
                                                 num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'])
        val_images_op, val_labels_op = input_data.inputs_images(filenames = val_files,
                                                   batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                   num_epochs=None,
                                                   num_threads=cfg['DATA']['NUM_THREADS'],
                                                   num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'])

        conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        conf.gpu_options.allow_growth = True
        sess = tf.Session(config=conf)
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if start_from_trained_model:
            vardict = {v.name[18:-2]: v for v in tf.trainable_variables()}
            saver = tf.train.Saver(vardict)
            print(vardict)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.degradation_models)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session Restored!')
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.degradation_models)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir+'train', sess.graph)
        val_writer = tf.summary.FileWriter(FLAGS.log_dir+'val', sess.graph)

        print("Training...")
        saver = tf.train.Saver(tf.trainable_variables())
        for step in range(cfg['TRAIN']['MAX_STEPS']):
            # Run by batch images
            start_time = time.time()
            tr_images, tr_labels = sess.run([tr_images_op, tr_labels_op])
            print(tr_images.shape)
            print(tr_labels.shape)
            tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
            _, loss_value = sess.run([train_op, loss_op], feed_dict=tr_feed)
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            print("Step: [%2d], time: [%4.4f], training_loss = [%.8f]" % (step, time.time()-start_time, loss_value))
            if step % cfg['TRAIN']['VAL_STEP'] == 0:
                val_images, val_labels = sess.run([val_images_op, val_labels_op])
                val_feed = {images_placeholder: val_images, labels_placeholder: val_labels}
                summary, loss_value, psnr = sess.run([merged, loss_op, psnr_op], feed_dict=val_feed)
                print("Step: [%2d], time: [%4.4f], validation_loss = [%.8f], validation_psnr = [%.8f]" %
                      (step, time.time()-start_time, loss_value, psnr))
                val_writer.add_summary(summary, step)
                tr_images, tr_labels = sess.run([tr_images_op, tr_labels_op])
                tr_feed = {images_placeholder: tr_images, labels_placeholder: tr_labels}
                summary, loss_value, psnr = sess.run([merged, loss_op, psnr_op], feed_dict=tr_feed)
                print("Step: [%2d], time: [%4.4f], training_loss = [%.8f], training_psnr = [%.8f]" %
                      (step, time.time()-start_time, loss_value, psnr))
                train_writer.add_summary(summary, step)
            if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step+1) == cfg['TRAIN']['MAX_STEPS']:
                checkpoint_path = os.path.join(FLAGS.degradation_models, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        coord.request_stop()
        coord.join(threads)
        sess.close()

# Initialize UtilityNet (fT) with fixed DegradNet (fd)
def run_pretraining(cfg):
    # Create model directory
    if not os.path.exists(FLAGS.whole_pretraining):
        os.makedirs(FLAGS.whole_pretraining)
    if not os.path.exists(FLAGS.eval_vis_dir):
        os.makedirs(FLAGS.eval_vis_dir)

    use_pretrained_model = True
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True


    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = placeholder_inputs(cfg, cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'])
            eval_video_placeholder = tf.placeholder(tf.float32, shape=(1, cfg['DATA']['DEPTH'], 120, 160, cfg['DATA']['NCHANNEL']))

            tower_grads_degradation, tower_grads_utility_main, tower_grads_utility_finetune = [], [], []

            # Compute Acc
            logits_utility_lst, logits_budget_lst = [], []

            # Compute Loss
            loss_utility_lst, loss_budget_lst = [], []

            opt_degradation = tf.train.AdamOptimizer(1e-3)
            opt_utility_finetune = tf.train.AdamOptimizer(1e-4)
            opt_utility = tf.train.AdamOptimizer(1e-5)

            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            degrad_videos = residualNet(videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], is_video=True)
                            if gpu_index == 0:
                                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                                    video = tf.slice(eval_video_placeholder, [0, 0, 4, 24, 0],
                                                     [1, cfg['DATA']['DEPTH'], cfg['DATA']['CROP_HEIGHT'], cfg['DATA']['CROP_WIDTH'], cfg['DATA']['NCHANNEL']])
                                    eval_video = residualNet(video, is_video=True)

                            degrad_videos = _avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos
                            logits_utility = utilityNet(degrad_videos, dropout_placeholder, wd=0.001)
                            logits_utility_lst.append(logits_utility)
                            loss_utility = tower_loss_xentropy_sparse(
                                scope,
                                logits_utility,
                                utility_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']],
                                use_weight_decay = True
                                )
                            loss_utility_lst.append(loss_utility)

                            varlist_degradtion = [v for v in tf.trainable_variables() if any(x in v.name for x in ["DegradationModule"])]
                            print([v.name for v in varlist_degradtion])
                            varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
                            print([v.name for v in varlist_utility])
                            varlist_utility_finetune = [v for v in varlist_utility if any(x in v.name.split('/')[1] for x in ["out", "d2"])]
                            print([v.name for v in varlist_utility_finetune])
                            varlist_utility_main = list(set(varlist_utility) - set(varlist_utility_finetune))
                            print([v.name for v in varlist_utility_main])

                            grads_degradation = opt_degradation.compute_gradients(loss_utility, varlist_degradtion)
                            grads_utility_main = opt_utility.compute_gradients(loss_utility, varlist_utility_main)
                            grads_utility_finetune = opt_utility_finetune.compute_gradients(loss_utility, varlist_utility_finetune)

                            tower_grads_degradation.append(grads_degradation)
                            tower_grads_utility_main.append(grads_utility_main)
                            tower_grads_utility_finetune.append(grads_utility_finetune)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

            eval_video_op = eval_video

            loss_utility_op = tf.reduce_mean(loss_utility_lst, name='softmax')

            logits_utility_op = tf.concat(logits_utility_lst, 0)
            accuracy_util = accuracy(logits_utility_op, utility_labels_placeholder)

            grads_degradation = average_gradients(tower_grads_degradation)
            grads_utility_main = average_gradients(tower_grads_utility_main)
            grads_utility_finetune = average_gradients(tower_grads_utility_finetune)

            with tf.device('/cpu:%d' % 0):
                tvs_degradation = varlist_degradtion
                accum_vars_degradtion =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_degradation]
                zero_ops_degradation = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_degradtion]

            with tf.device('/cpu:%d' % 0):
                tvs_utility_finetune = varlist_utility_finetune
                accum_vars_utility_finetune =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility_finetune]
                zero_ops_utility_finetune = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility_finetune]

            with tf.device('/cpu:%d' % 0):
                tvs_utility_main = varlist_utility_main
                accum_vars_utility_main =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility_main]
                zero_ops_utility_main = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility_main]

            accum_ops_degradation = [accum_vars_degradtion[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                         enumerate(grads_degradation)]
            accum_ops_utility_main = [accum_vars_utility_main[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv
                                          in enumerate(grads_utility_main)]
            accum_ops_utility_finetune = [accum_vars_utility_finetune[i].assign_add(gv[0] / FLAGS.n_minibatches) for
                                              i, gv in enumerate(grads_utility_finetune)]

            apply_gradient_op_degradation = opt_degradation.apply_gradients(
                    [(accum_vars_degradtion[i].value(), gv[1]) for i, gv in enumerate(grads_degradation)],
                    global_step=global_step)
            apply_gradient_op_utility_main = opt_utility.apply_gradients(
                    [(accum_vars_utility_main[i].value(), gv[1]) for i, gv in enumerate(grads_utility_main)],
                    global_step=global_step)
            apply_gradient_op_utility_finetune = opt_utility.apply_gradients(
                    [(accum_vars_utility_finetune[i].value(), gv[1]) for i, gv in enumerate(grads_utility_finetune)],
                    global_step=global_step)

            train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for f in
                               os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
            val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for f in
                             os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]
            test_files = [os.path.join(cfg['DATA']['TEST_FILES_DIR'], f) for f in
                              os.listdir(cfg['DATA']['TEST_FILES_DIR']) if f.endswith('.tfrecords')]

            print(train_files)
            print(val_files)
            print(test_files)

            tr_videos_op, tr_videos_labels_op, _ = input_data.inputs_videos(filenames=train_files,
                                                                             batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                                             num_epochs=None,
                                                                             num_threads=cfg['DATA']['NUM_THREADS'],
                                                                             num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                                             shuffle=True)
            val_videos_op, val_videos_labels_op, _ = input_data.inputs_videos(filenames=val_files,
                                                                               batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                                               num_epochs=None,
                                                                               num_threads=cfg['DATA']['NUM_THREADS'],
                                                                               num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                                               shuffle=True)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            # Create a saver for writing training checkpoints.
            if use_pretrained_model:
                varlist = [v for v in tf.trainable_variables() if not any(x in v.name.split('/')[0] for x in ["UtilityModule", "BudgetModule"])]
                restore_model_ckpt(sess, FLAGS.degradation_models, varlist, "DegradationModule")
                restore_model_pretrained_C3D(sess, cfg)
            else:
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.whole_pretraining)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.whole_pretraining)

            saver = tf.train.Saver(tf.trainable_variables())
            eval_video = np.load('../eval_video.npy')
            eval_video = np.reshape(eval_video, [1, 16, 120, 160, 3])
            for step in xrange(500):
                eval_video_output = np.squeeze(
                    sess.run(eval_video_op, feed_dict={eval_video_placeholder: eval_video}))
                print(eval_video_output.shape)
                plot_visualization_frame(FLAGS.eval_vis_dir, eval_video_output[6], step)
                start_time = time.time()
                sess.run([zero_ops_utility_finetune, zero_ops_utility_main, zero_ops_degradation])
                loss_utility_lst = []
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_videos_labels = sess.run([tr_videos_op, tr_videos_labels_op])
                    _, _, loss_utility = sess.run([accum_ops_utility_finetune, accum_ops_utility_main, loss_utility_op],
                                                         feed_dict={videos_placeholder: tr_videos,
                                                                    utility_labels_placeholder: tr_videos_labels,
                                                                    dropout_placeholder: 1.0,
                                                                    })
                    loss_utility_lst.append(loss_utility)
                sess.run([apply_gradient_op_utility_finetune, apply_gradient_op_utility_main,
                                  apply_gradient_op_degradation])
                loss_summary = 'Utility Module + Degradation Module, Step: {:4d}, time: {:.4f}, utility loss: {:.8f}'.format(
                            step,
                            time.time() - start_time,
                            np.mean(loss_utility_lst))
                print(loss_summary)

                if step % cfg['TRAIN']['VAL_STEP'] == 0:
                    start_time = time.time()
                    acc_util_train_lst, loss_utility_train_lst = [], []
                    for _ in itertools.repeat(None, 30):
                        tr_videos, tr_videos_labels = sess.run([tr_videos_op, tr_videos_labels_op])
                        acc_util, loss_utility = sess.run([accuracy_util, loss_utility_op],
                                feed_dict={videos_placeholder: tr_videos,
                                           utility_labels_placeholder: tr_videos_labels,
                                           dropout_placeholder: 1.0,
                                           })
                        acc_util_train_lst.append(acc_util)
                        loss_utility_train_lst.append(loss_utility)

                    train_summary = "Step: {:4d}, time: {:.4f}, utility loss: {:.8f}, training utility accuracy: {:.5f}".format(
                            step,
                            time.time() - start_time,
                            np.mean(loss_utility_train_lst), np.mean(acc_util_train_lst))
                    print(train_summary)

                    start_time = time.time()
                    acc_util_val_lst, loss_utility_val_lst = [], []
                    for _ in itertools.repeat(None, 30):
                        val_videos, val_videos_labels = sess.run([val_videos_op, val_videos_labels_op])
                        acc_util, loss_utility = sess.run([accuracy_util, loss_utility_op],
                                feed_dict={videos_placeholder: val_videos,
                                           utility_labels_placeholder: val_videos_labels,
                                           dropout_placeholder: 1.0,
                                           })
                        acc_util_val_lst.append(acc_util)
                        loss_utility_val_lst.append(loss_utility)

                    test_summary = "Step: {:4d}, time: {:.4f}, utility loss: {:.8f}, validation utility accuracy: {:.5f}".format(
                            step,
                            time.time() - start_time,
                            np.mean(loss_utility_val_lst), np.mean(acc_util_val_lst))
                    print(test_summary)

                if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == cfg['TRAIN']['MAX_STEPS']:
                    checkpoint_path = os.path.join(FLAGS.whole_pretraining, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)
        print("done")

# Using negative entropy loss for adversarial training
# Algorithm 1 in the paper
def run_training_ensembling_restarting_nentropy(cfg):
    # Create model directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.saved_checkpoint_dir):
        os.makedirs(FLAGS.saved_checkpoint_dir)
    if not os.path.exists(FLAGS.eval_vis_dir):
        os.makedirs(FLAGS.eval_vis_dir)

    use_pretrained_model = True
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    multiplier_lst = [0.60 - i * 0.02 for i in range(FLAGS.NBudget)]
    from collections import defaultdict
    logits_budget_lst_dct = defaultdict(list)
    loss_budget_lst_dct = defaultdict(list)
    wplaceholder_dct = {}

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = placeholder_inputs(cfg, cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'])
            eval_video_placeholder = tf.placeholder(tf.float32, shape=(1, cfg['DATA']['DEPTH'], 120, 160, cfg['DATA']['NCHANNEL']))
            isTraining_placeholder = tf.placeholder(tf.bool)
            isAdversarial_placeholder = tf.placeholder(tf.bool)
            for multiplier in multiplier_lst:
                wplaceholder_dct['{}'.format(multiplier)] = tf.placeholder(tf.float32)

            tower_grads_degradation, tower_grads_utility, tower_grads_budget = [], [], []

            # Compute Acc
            logits_utility_lst, logits_budget_lst = [], []

            # Compute Loss
            loss_utility_lst, loss_budget_lst, loss_degrad_lst = [], [], []

            # Compute prediction with min entropy (most confident)
            # Use max uniform loss instead
            argmax_entropy_lst = []

            # Optimizer for the 3 components respectively
            opt_degradation = tf.train.AdamOptimizer(FLAGS.degradation_lr)
            opt_utility = tf.train.AdamOptimizer(FLAGS.utility_lr)
            opt_budget = tf.train.AdamOptimizer(FLAGS.budget_lr)

            if FLAGS.use_lambda_decay:
                _lambda_op = tf.train.exponential_decay(FLAGS._lambda, global_step=global_step, decay_steps=10, decay_rate=0.9)
            else:
                _lambda_op = tf.identity(FLAGS._lambda)

            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            degrad_videos = residualNet(videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], is_video=True)
                            L1_loss = tf.reduce_mean(tf.abs(degrad_videos - videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]))
                            if gpu_index == 0:
                                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                                    video = tf.slice(eval_video_placeholder, [0, 0, 4, 24, 0],
                                                     [1, cfg['DATA']['DEPTH'], cfg['DATA']['CROP_HEIGHT'], cfg['DATA']['CROP_WIDTH'], cfg['DATA']['NCHANNEL']])
                                    eval_video = residualNet(video, is_video=True)

                            degrad_videos = _instance_norm(degrad_videos) if cfg['DATA_PROCESSING']['USE_INSTANCE_NORM'] else degrad_videos
                            degrad_videos = _bilinear_resize(degrad_videos) if cfg['DATA_PROCESSING']['USE_BILINEAR_RESIZE'] else degrad_videos
                            degrad_videos = tf.cond(isAdversarial_placeholder, lambda: tf.identity(degrad_videos), lambda: _binary_activation(degrad_videos)) \
                                                                                                    if cfg['DATA_PROCESSING']['USE_BINARIZE'] else degrad_videos
                            degrad_videos = _avg_replicate(degrad_videos) if FLAGS.use_avg_replicate else degrad_videos

                            logits_utility = utilityNet(degrad_videos, dropout_placeholder, wd=0.001)
                            logits_utility_lst.append(logits_utility)
                            loss_utility = tower_loss_xentropy_sparse(scope, logits_utility, utility_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], use_weight_decay = True)
                            loss_utility_lst.append(loss_utility)
                            logits_budget = tf.zeros([cfg['TRAIN']['BATCH_SIZE'], cfg['DATA']['NUM_CLASSES_BUDGET']])
                            loss_budget = 0.0
                            loss_budget_entropy= 0.0
                            weighted_loss_budget_entropy = 0.0
                            logits_lst = []
                            for multiplier in multiplier_lst:
                                print(multiplier)
                                logits = budgetNet(degrad_videos, isTraining_placeholder, depth_multiplier=multiplier)
                                logits_lst.append(logits)
                                loss = tower_loss_xentropy_sparse(scope, logits, budget_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']],
                                                                  use_weight_decay = False)
                                loss_neg_entropy = tower_loss_neg_entropy(logits)
                                logits_budget_lst_dct[str(multiplier)].append(logits)
                                loss_budget_lst_dct[str(multiplier)].append(loss)
                                logits_budget += logits / FLAGS.NBudget
                                loss_budget += loss / FLAGS.NBudget

                                weighted_loss_budget_entropy += wplaceholder_dct[str(multiplier)] * loss_neg_entropy
                                loss_budget_entropy += loss_neg_entropy / FLAGS.NBudget

                            max_nentropy, argmax_nentropy = tower_loss_max_neg_entropy(logits_lst)
                            logits_budget_lst.append(logits_budget)
                            loss_budget_lst.append(loss_budget)
                            argmax_entropy_lst.append(argmax_nentropy)

                            if FLAGS.mode == 'SuppressingMostConfident':
                                if FLAGS.use_l1_loss:
                                    loss_degrad = loss_utility + FLAGS._gamma * max_nentropy + _lambda_op * L1_loss
                                else:
                                    loss_degrad = loss_utility + FLAGS._gamma_ * max_nentropy
                            elif FLAGS.mode == 'Batch':
                                loss_degrad = loss_utility + FLAGS._gamma_ * loss_budget_entropy
                            elif FLAGS.mode == 'Online':
                                loss_degrad = loss_utility + FLAGS._gamma_ * weighted_loss_budget_entropy
                            else:
                                raise ValueError("Wrong given mode")
                            loss_degrad_lst.append(loss_degrad)

                            varlist_degradtion = [v for v in tf.trainable_variables() if any(x in v.name for x in ["DegradationModule"])]
                            print("####################################################DegradationModuleVariables####################################################")
                            print([v.name for v in varlist_degradtion])
                            varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
                            print("####################################################UtilityModuleVariables####################################################")
                            print([v.name for v in varlist_utility])
                            varlist_budget = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]
                            print("####################################################BudgetModuleVariables####################################################")
                            print([v.name for v in varlist_budget])


                            grads_degradation = opt_degradation.compute_gradients(loss_degrad, varlist_degradtion)
                            grads_budget = opt_budget.compute_gradients(loss_budget, varlist_budget)
                            grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility+varlist_degradtion)

                            tower_grads_degradation.append(grads_degradation)
                            tower_grads_budget.append(grads_budget)
                            tower_grads_utility.append(grads_utility)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
            eval_video_op = eval_video
            argmax_ent_op = tf.concat(argmax_entropy_lst, 0)
            loss_utility_op = tf.reduce_mean(loss_utility_lst, name='softmax')
            tf.summary.scalar('utility loss', loss_utility_op)
            loss_budget_op = tf.reduce_mean(loss_budget_lst, name='softmax')
            tf.summary.scalar('budget loss', loss_budget_op)
            loss_degrad_op = tf.reduce_mean(loss_degrad_lst, name='softmax')
            tf.summary.scalar('degrad loss', loss_degrad_op)

            logits_utility = tf.concat(logits_utility_lst, 0)
            accuracy_util = accuracy(logits_utility, utility_labels_placeholder)
            tf.summary.scalar('utility task accuracy', accuracy_util)

            logits_budget = tf.concat(logits_budget_lst, 0)
            accuracy_budget = accuracy(logits_budget, budget_labels_placeholder)
            tf.summary.scalar('budget task average accuracy', accuracy_budget)

            acc_op_lst = []
            loss_op_lst = []
            for multiplier in multiplier_lst:
                acc_op = accuracy(tf.concat(logits_budget_lst_dct[str(multiplier)], 0), budget_labels_placeholder)
                acc_op_lst.append(acc_op)
                tf.summary.scalar('budget module {} accuracy'.format(multiplier), acc_op)
                loss_op = tf.reduce_max(loss_budget_lst_dct[str(multiplier)])
                loss_op_lst.append(loss_op)
                tf.summary.scalar('budget module {} loss'.format(multiplier), loss_op)

            grads_degradation = average_gradients(tower_grads_degradation)
            grads_budget = average_gradients(tower_grads_budget)
            grads_utility = average_gradients(tower_grads_utility)

            with tf.device('/cpu:%d' % 0):
                tvs_degradation = varlist_degradtion
                accum_vars_degradtion =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_degradation]
                zero_ops_degradation = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_degradtion]

            with tf.device('/cpu:%d' % 0):
                tvs_budget = varlist_budget
                accum_vars_budget =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_budget]
                zero_ops_budget = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_budget]

            with tf.device('/cpu:%d' % 0):
                tvs_utility = varlist_utility + varlist_degradtion
                accum_vars_utility =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility]
                zero_ops_utility = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility]

            #zero_ops = zero_ops_degradation + zero_ops_budget + zero_ops_utility
            global_increment = global_step.assign_add(1)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            # print(tf.get_collection_ref(tf.GraphKeys.MOVING_AVERAGE_VARIABLES))
            #with tf.control_dependencies([tf.group(*update_ops), add_global]):
            with tf.control_dependencies([tf.no_op('update_op')]):
                accum_ops_degradation = [accum_vars_degradtion[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                         enumerate(grads_degradation)]
                accum_ops_utility = [accum_vars_utility[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                     enumerate(grads_utility)]
                accum_ops_budget = [accum_vars_budget[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                    enumerate(grads_budget)]


            #accum_ops = accum_ops_degradation + accum_ops_utility + accum_ops_budget
            with tf.control_dependencies([global_increment]):
                apply_gradient_op_degradation = opt_degradation.apply_gradients([(accum_vars_degradtion[i].value(), gv[1]) for i, gv in enumerate(grads_degradation)], global_step=None)

            apply_gradient_op_utility = opt_utility.apply_gradients([(accum_vars_utility[i].value(), gv[1]) for i, gv in enumerate(grads_utility)], global_step=None)
            apply_gradient_op_budget = opt_budget.apply_gradients([(accum_vars_budget[i].value(), gv[1]) for i, gv in enumerate(grads_budget)], global_step=None)

            train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for f in
                           os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
            val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for f in
                         os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]

            print(train_files)
            print(val_files)

            tr_videos_op, tr_action_labels_op, tr_actor_labels_op = input_data.inputs_videos(filenames = train_files,
                                                 batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                 num_epochs=None,
                                                 num_threads=cfg['DATA']['NUM_THREADS'],
                                                 num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                 shuffle=True)
            val_videos_op, val_action_labels_op, val_actor_labels_op = input_data.inputs_videos(filenames = val_files,
                                                   batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                   num_epochs=None,
                                                   num_threads=cfg['DATA']['NUM_THREADS'],
                                                   num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                   shuffle=True)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            if use_pretrained_model:
                vardict_degradation = {v.name[18:-2]: v for v in varlist_degradtion}
                vardict_utility = {v.name[:-2]: v for v in varlist_utility}
                vardict = dict(vardict_degradation, **vardict_utility)
                restore_model_ckpt(sess, FLAGS.whole_pretraining, vardict, "DegradationModule")
            else:
                saver = tf.train.Saver(tf.trainable_variables())
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)
            ckpt_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=20)

            if not os.path.exists(FLAGS.summary_dir):
                os.makedirs(FLAGS.summary_dir)
            loss_summary_file = open(FLAGS.summary_dir+'loss_summary.txt', 'w')
            train_summary_file = open(FLAGS.summary_dir+'train_summary.txt', 'w')
            test_summary_file = open(FLAGS.summary_dir+'test_summary.txt', 'w')
            model_sampling_summary_file = open(FLAGS.summary_dir+'model_summary.txt', 'w')

            largest_gap = 0.15

            eval_video = np.load('../eval_video.npy')
            eval_video = np.reshape(eval_video, [1, 16, 120, 160, 3])
            for step in xrange(cfg['TRAIN']['MAX_STEPS']):
                if step == 0 or (FLAGS.use_resampling and step % FLAGS.resample_step == 0):
                    budget_varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]
                    init_budget_op = tf.variables_initializer(budget_varlist)
                    sess.run(init_budget_op)
                    for _ in itertools.repeat(None, FLAGS.retraining_step):
                        start_time = time.time()
                        acc_util_lst, acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        sess.run(zero_ops_budget)
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            tr_videos, tr_action_labels, tr_actor_labels = sess.run(
                                [tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                            _, acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run(
                                [accum_ops_budget, accuracy_util, accuracy_budget, loss_degrad_op,
                                 loss_utility_op, loss_budget_op],
                                feed_dict={videos_placeholder: tr_videos,
                                           utility_labels_placeholder: tr_action_labels,
                                           budget_labels_placeholder: tr_actor_labels,
                                           dropout_placeholder: 1.0,
                                           isTraining_placeholder: True,
                                           isAdversarial_placeholder:False,
                                           })
                            acc_util_lst.append(acc_util)
                            acc_budget_lst.append(acc_budget)
                            loss_degrad_lst.append(loss_degrad_value)
                            loss_utility_lst.append(loss_utility_value)
                            loss_budget_lst.append(loss_budget_value)
                        sess.run(apply_gradient_op_budget)
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Resampling (Budget), Step: {:4d}, time: {:.4f}, budget loss: {:.8f}, training budget accuracy: {:.5f}, ' \
                                       'utility loss: {:.8f}, training utility accuracy: {:.5f}'.format(step,
                                        time.time() - start_time, np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(loss_utility_lst), np.mean(acc_util_lst))
                        model_sampling_summary_file.write(loss_summary + '\n')
                        print(loss_summary)
                start_time = time.time()
                loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], []
                sess.run(zero_ops_degradation)
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_actor_labels, tr_action_labels = sess.run(
                                    [tr_videos_op, tr_actor_labels_op, tr_action_labels_op])
                    _, argmax_cent, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_degradation,
                                                    argmax_ent_op, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={videos_placeholder: tr_videos,
                                                               utility_labels_placeholder: tr_action_labels,
                                                               budget_labels_placeholder: tr_actor_labels,
                                                               dropout_placeholder: 1.0,
                                                               isTraining_placeholder: True,
                                                               isAdversarial_placeholder:True,
                                                               })
                    print(argmax_cent)
                    #print(loss_budget_uniform_tensor)
                    loss_degrad_lst.append(loss_degrad_value)
                    loss_utility_lst.append(loss_utility_value)
                    loss_budget_lst.append(loss_budget_value)
                _, _lambda = sess.run([apply_gradient_op_degradation, _lambda_op])

                assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                duration = time.time() - start_time


                loss_summary = 'Alternating Training (Degradation), Lambda: {:.8f}, Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}'.format(_lambda, step,
                                                        duration, np.mean(loss_degrad_lst), np.mean(loss_utility_lst), np.mean(loss_budget_lst))

                print(loss_summary)
                loss_summary_file.write(loss_summary + '\n')

                if FLAGS.use_monitor_utility:
                    while True:
                        start_time = time.time()
                        acc_util_lst, acc_budget_lst, loss_value_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            val_videos, val_action_labels, val_actor_labels = sess.run([val_videos_op, val_action_labels_op, val_actor_labels_op])
                            acc_util, acc_budget, loss_value, loss_utility, loss_budget = sess.run(
                                            [accuracy_util, accuracy_budget, loss_op, loss_utility_op, loss_budget_op],
                                                feed_dict={videos_placeholder: val_videos,
                                                    utility_labels_placeholder: val_action_labels,
                                                    budget_labels_placeholder: val_actor_labels,
                                                    dropout_placeholder: 1.0,
                                                    isTraining_placeholder: True,
                                                    isAdversarial_placeholder: True,
                                                })
                            acc_util_lst.append(acc_util)
                            acc_budget_lst.append(acc_budget)
                            loss_value_lst.append(loss_value)
                            loss_utility_lst.append(loss_utility)
                            loss_budget_lst.append(loss_budget)
                            # test_writer.add_summary(summary, step)
                        val_summary = "Step: {:4d}, time: {:.4f}, total loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f},\n" .format(
                                step,
                                time.time() - start_time, np.mean(loss_value_lst),
                                np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                np.mean(acc_util_lst), np.mean(acc_budget_lst))
                        print(val_summary)

                        if np.mean(acc_util_lst) >= FLAGS.highest_util_acc_val:
                            break
                        start_time = time.time()
                        sess.run(zero_ops_utility)
                        acc_util_lst, acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                            _, acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_utility, accuracy_util, accuracy_budget, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={videos_placeholder: tr_videos,
                                                               utility_labels_placeholder: tr_action_labels,
                                                               budget_labels_placeholder: tr_actor_labels,
                                                               dropout_placeholder: 0.5,
                                                               isTraining_placeholder: True,
                                                               isAdversarial_placeholder: True,
                                                               })
                            acc_util_lst.append(acc_util)
                            acc_budget_lst.append(acc_budget)
                            loss_degrad_lst.append(loss_degrad_value)
                            loss_utility_lst.append(loss_utility_value)
                            loss_budget_lst.append(loss_budget_value)
                        sess.run([apply_gradient_op_utility])
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Utility), Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, '.format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst))

                        print(loss_summary)

                if FLAGS.use_monitor_budget:
                    while True:
                        start_time = time.time()
                        sess.run(zero_ops_budget)
                        acc_util_lst, acc_budget_lst, loss_degrad_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])

                            _, acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_budget, accuracy_util, accuracy_budget, loss_degrad_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={videos_placeholder: tr_videos,
                                                               utility_labels_placeholder: tr_action_labels,
                                                               budget_labels_placeholder: tr_actor_labels,
                                                               dropout_placeholder: 1.0,
                                                               isTraining_placeholder: True,
                                                               isAdversarial_placeholder: True,
                                                               })
                            acc_util_lst.append(acc_util)
                            acc_budget_lst.append(acc_budget)
                            loss_degrad_lst.append(loss_degrad_value)
                            loss_utility_lst.append(loss_utility_value)
                            loss_budget_lst.append(loss_budget_value)
                        sess.run([apply_gradient_op_budget])
                        assert not np.isnan(np.mean(loss_degrad_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Budget), Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, ' \
                                   'training utility accuracy: {:.5f}, training budget accuracy: {:.5f}'.format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                                            np.mean(acc_util_lst), np.mean(acc_budget_lst))

                        print(loss_summary)
                        if np.mean(acc_budget_lst) >= FLAGS.highest_budget_acc_train:
                            break

                if step % cfg['TRAIN']['VAL_STEP'] == 0:
                    eval_video_output = np.squeeze(
                        sess.run(eval_video_op, feed_dict={eval_video_placeholder: eval_video}))
                    print(eval_video_output.shape)
                    plot_visualization_frame(eval_video_output[6], step)
                    start_time = time.time()
                    acc_util_train_lst, acc_budget_train_lst, loss_degrad_train_lst, loss_utility_train_lst, loss_budget_train_lst = [], [], [], [], []
                    acc_util_val_lst, acc_budget_val_lst, loss_degrad_val_lst, loss_utility_val_lst, loss_budget_val_lst = [], [], [], [], []
                    for _ in itertools.repeat(None, FLAGS.n_minibatches):
                        tr_videos, tr_action_labels, tr_actor_labels = sess.run(
                                    [tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                        acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accuracy_util, accuracy_budget,
                                                                                                loss_degrad_op, loss_utility_op, loss_budget_op],
                                                           feed_dict={videos_placeholder: tr_videos,
                                                                      utility_labels_placeholder: tr_action_labels,
                                                                      budget_labels_placeholder: tr_actor_labels,
                                                                      dropout_placeholder: 1.0,
                                                                      isTraining_placeholder: True,
                                                                      isAdversarial_placeholder: True,
                                                                      })
                        acc_util_train_lst.append(acc_util)
                        acc_budget_train_lst.append(acc_budget)
                        loss_degrad_train_lst.append(loss_degrad_value)
                        loss_utility_train_lst.append(loss_utility_value)
                        loss_budget_train_lst.append(loss_budget_value)

                    train_summary = "Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, training utility accuracy: {:.5f}, training budget accuracy: {:.5f}".format(
                            step,
                            time.time() - start_time, np.mean(loss_degrad_train_lst),
                            np.mean(loss_utility_train_lst), np.mean(loss_budget_train_lst),
                            np.mean(acc_util_train_lst), np.mean(acc_budget_train_lst))
                    print(train_summary)
                    train_summary_file.write(train_summary + '\n')

                    for _ in itertools.repeat(None, FLAGS.n_minibatches_eval):
                        val_videos, val_action_labels, val_actor_labels = sess.run(
                                        [val_videos_op, val_action_labels_op, val_actor_labels_op])
                        acc_util, acc_budget, loss_degrad_value, loss_utility_value, loss_budget_value = sess.run([accuracy_util, accuracy_budget,
                                                                                                loss_degrad_op, loss_utility_op, loss_budget_op],
                                                            feed_dict={videos_placeholder: val_videos,
                                                                        utility_labels_placeholder: val_action_labels,
                                                                        budget_labels_placeholder: val_actor_labels,
                                                                        dropout_placeholder: 1.0,
                                                                        isTraining_placeholder: True,
                                                                        isAdversarial_placeholder: True,
                                                                       })
                        acc_util_val_lst.append(acc_util)
                        acc_budget_val_lst.append(acc_budget)
                        loss_degrad_val_lst.append(loss_degrad_value)
                        loss_utility_val_lst.append(loss_utility_value)
                        loss_budget_val_lst.append(loss_budget_value)

                    test_summary = "Step: {:4d}, time: {:.4f}, degrad loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f}".format(step,
                                                            time.time() - start_time, np.mean(loss_degrad_val_lst),
                                                            np.mean(loss_utility_val_lst), np.mean(loss_budget_val_lst),
                                                            np.mean(acc_util_val_lst), np.mean(acc_budget_val_lst))
                    print(test_summary)
                    test_summary_file.write(test_summary + '\n')

                gap = np.mean(acc_util_val_lst) - np.mean(acc_budget_val_lst)

                if gap > largest_gap and step % FLAGS.resample_step != 0:
                    largest_gap = gap
                    checkpoint_path = os.path.join(FLAGS.saved_checkpoint_dir, '{}_model.ckpt'.format(gap))
                    ckpt_saver.save(sess, checkpoint_path, global_step=step)

                if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == cfg['TRAIN']['MAX_STEPS']:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            loss_summary_file.close()
            train_summary_file.close()
            test_summary_file.close()
            coord.request_stop()
            coord.join(threads)

    print("done")

# Using cross entropy uniform loss for adversarial training
# Algorithm 1 in the paper
def run_training_model_sampling(cfg):
    # Create model directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.saved_checkpoint_dir):
        os.makedirs(FLAGS.saved_checkpoint_dir)
    if not os.path.exists(FLAGS.eval_vis_dir):
        os.makedirs(FLAGS.eval_vis_dir)

    use_pretrained_model = True
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    multiplier_lst = [0.58 - i * 0.02 for i in range(FLAGS.NBudget)]
    from collections import defaultdict
    logits_budget_lst_dct = defaultdict(list)
    loss_budget_lst_dct = defaultdict(list)
    wplaceholder_dct = {}

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = placeholder_inputs(cfg, cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'])
            eval_video_placeholder = tf.placeholder(tf.float32, shape=(1, cfg['DATA']['DEPTH'], 120, 160, cfg['DATA']['NCHANNEL']))
            budget_uniform_labels_placeholder = tf.placeholder(tf.float32, shape=(cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'], cfg['DATA']['NUM_CLASSES_BUDGET']))
            isTraining_placeholder = tf.placeholder(tf.bool)
            isAdversarial_placeholder = tf.placeholder(tf.bool)
            for multiplier in multiplier_lst:
                wplaceholder_dct[str(multiplier)] = tf.placeholder(tf.float32)

            tower_grads_degradation = []
            tower_grads_utility = []
            tower_grads_utility_only = []
            tower_grads_budget = []

            # Compute Acc
            logits_utility_lst = []
            logits_budget_lst = []

            # Compute Loss
            loss_utility_lst = []
            loss_budget_lst = []
            loss_filter_lst = []

            # Compute prediction with min entropy (most confident)
            # Use max uniform loss instead
            min_centpy_lst = []
            argmax_centpy_lst = []
            loss_budget_uniform_tensor_lst = []

            opt_degradation = tf.train.AdamOptimizer(FLAGS.degradation_lr)
            opt_utility = tf.train.AdamOptimizer(FLAGS.utility_lr)
            opt_budget = tf.train.AdamOptimizer(FLAGS.budget_lr)

            if FLAGS.use_lambda_decay:
                _lambda_op = tf.train.exponential_decay(FLAGS._lambda, global_step=global_step, decay_steps=10, decay_rate=0.9)
            else:
                _lambda_op = tf.identity(FLAGS._lambda)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            degrad_videos = residualNet(videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], is_video=True)
                            L1_loss = tf.reduce_mean(tf.abs(degrad_videos - videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]))
                            if gpu_index == 0:
                                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                                    video = tf.slice(eval_video_placeholder, [0, 0, 4, 24, 0],
                                                     [1, cfg['DATA']['DEPTH'], cfg['DATA']['CROP_HEIGHT'], cfg['DATA']['CROP_WIDTH'], cfg['DATA']['NCHANNEL']])
                                    eval_video = residualNet(video, is_video=True)
                            if cfg['DATA_PROCESSING']['USE_INSTANCE_NORM']:
                                degrad_videos = _instance_norm(degrad_videos)
                            if cfg['DATA_PROCESSING']['USE_BILINEAR_RESIZE']:
                                degrad_videos = _bilinear_resize(degrad_videos)
                            if cfg['DATA_PROCESSING']['USE_BINARIZE']:
                                degrad_videos = tf.cond(isAdversarial_placeholder, lambda: tf.identity(degrad_videos), lambda: _binary_activation(degrad_videos))
                            if FLAGS.use_avg_replicate:
                                degrad_videos = _avg_replicate(degrad_videos)

                            logits_utility = utilityNet(degrad_videos, dropout_placeholder, wd=0.001)
                            logits_utility_lst.append(logits_utility)
                            loss_utility = tower_loss_xentropy_sparse(
                                scope,
                                logits_utility,
                                utility_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']],
                                use_weight_decay = True
                                )
                            loss_utility_lst.append(loss_utility)
                            logits_budget = tf.zeros([cfg['TRAIN']['BATCH_SIZE'], cfg['DATA']['NUM_CLASSES_BUDGET']])
                            loss_budget = 0.0
                            loss_budget_uniform = 0.0
                            weighted_loss_budget_uniform = 0.0
                            loss_uniform_tensor_lst = []
                            for multiplier in multiplier_lst:
                                logits = budgetNet(degrad_videos, isTraining_placeholder, depth_multiplier=multiplier)
                                loss = tower_loss_xentropy_sparse(
                                    scope,
                                    logits,
                                    budget_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']],
                                    use_weight_decay = False
                                )
                                loss_uniform = tower_loss_xentropy_dense(
                                    logits,
                                    budget_uniform_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE'], :]
                                )
                                loss_uniform_tensor = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=budget_uniform_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE'],:])
                                print(multiplier)
                                print(loss_uniform_tensor)
                                print('##################################################################')
                                logits_budget_lst_dct['{}'.format(multiplier)].append(logits)
                                loss_budget_lst_dct['{}'.format(multiplier)].append(loss)
                                logits_budget += logits
                                loss_budget += loss
                                loss_uniform_tensor_lst.append(loss_uniform_tensor)
                                weighted_loss_budget_uniform += wplaceholder_dct['{}'.format(multiplier)] * loss_uniform
                                loss_budget_uniform += loss_uniform
                            loss_budget_uniform_tensor_stack = tf.stack(loss_uniform_tensor_lst, axis=0)
                            print('##############################################################')
                            print(loss_budget_uniform_tensor_stack.shape)
                            print(tf.reduce_max(loss_budget_uniform_tensor_stack, axis=0).shape)
                            print('##############################################################')
                            argmax_centpy = tf.argmax(loss_budget_uniform_tensor_stack, axis=0)
                            min_centpy = tf.reduce_mean(tf.reduce_max(loss_budget_uniform_tensor_stack, axis=0))
                            logits_budget_lst.append(logits_budget)
                            loss_budget_lst.append(loss_budget)
                            min_centpy_lst.append(min_centpy)
                            argmax_centpy_lst.append(argmax_centpy)
                            loss_budget_uniform_tensor_lst.append(loss_budget_uniform_tensor_stack)

                            if FLAGS.mode == 'SuppressingMostConfident':
                                if FLAGS.use_l1_loss:
                                    loss_filter = loss_utility + FLAGS._gamma * min_centpy + _lambda_op * L1_loss
                                else:
                                    loss_filter = loss_utility + FLAGS._gamma_ * min_centpy
                            elif FLAGS.mode == 'Batch':
                                loss_filter = loss_utility + FLAGS._gamma_ * loss_budget_uniform
                            elif FLAGS.mode == 'Online':
                                loss_filter = loss_utility + FLAGS._gamma_ * weighted_loss_budget_uniform
                            else:
                                raise ValueError("Wrong given mode")
                            loss_filter_lst.append(loss_filter)

                            varlist_degradtion = [v for v in tf.trainable_variables() if any(x in v.name for x in ["DegradationModule"])]
                            print("####################################################DegradationModuleVariables####################################################")
                            print([v.name for v in varlist_degradtion])
                            varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["UtilityModule"])]
                            print("####################################################UtilityModuleVariables####################################################")
                            print([v.name for v in varlist_utility])
                            varlist_budget = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]
                            print("####################################################BudgetModuleVariables####################################################")
                            print([v.name for v in varlist_budget])


                            grads_degradation = opt_degradation.compute_gradients(loss_filter, varlist_degradtion)
                            grads_budget = opt_budget.compute_gradients(loss_budget, varlist_budget)
                            grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility+varlist_degradtion)
                            grads_utility_only = opt_utility.compute_gradients(loss_utility, varlist_utility)

                            tower_grads_degradation.append(grads_degradation)
                            tower_grads_budget.append(grads_budget)
                            tower_grads_utility.append(grads_utility)
                            tower_grads_utility_only.append(grads_utility_only)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
            eval_video_op = eval_video
            argmax_cent_op = tf.concat(argmax_centpy_lst, 0)
            loss_budget_uniform_tensor_op = tf.concat(loss_budget_uniform_tensor_lst, 1)
            loss_utility_op = tf.reduce_mean(loss_utility_lst, name='softmax')
            tf.summary.scalar('utility loss', loss_utility_op)
            loss_budget_op = tf.reduce_mean(loss_budget_lst, name='softmax')
            tf.summary.scalar('budget loss', loss_budget_op)
            loss_filter_op = tf.reduce_mean(loss_filter_lst, name='softmax')
            tf.summary.scalar('filtering loss', loss_filter_op)

            logits_utility = tf.concat(logits_utility_lst, 0)
            accuracy_util = accuracy(logits_utility, utility_labels_placeholder)
            tf.summary.scalar('utility task accuracy', accuracy_util)

            logits_budget = tf.concat(logits_budget_lst, 0)
            accuracy_budget = accuracy(logits_budget, budget_labels_placeholder)
            tf.summary.scalar('budget task average accuracy', accuracy_budget)

            acc_op_lst = []
            for multiplier in multiplier_lst:
                acc_op = accuracy(tf.concat(logits_budget_lst_dct['{}'.format(multiplier)], 0), budget_labels_placeholder)
                acc_op_lst.append(acc_op)
                tf.summary.scalar('budget module {} accuracy'.format(multiplier), acc_op)

            loss_op_lst = []
            for multiplier in multiplier_lst:
                loss_op = tf.reduce_max(loss_budget_lst_dct['{}'.format(multiplier)])
                loss_op_lst.append(loss_op)
                tf.summary.scalar('budget module {} loss'.format(multiplier), loss_op)

            grads_degradation = average_gradients(tower_grads_degradation)
            grads_budget = average_gradients(tower_grads_budget)
            grads_utility = average_gradients(tower_grads_utility)

            with tf.device('/cpu:%d' % 0):
                #tvs_degradation = varlist_degradtion+varlist_instance_norm
                tvs_degradation = varlist_degradtion
                accum_vars_degradtion =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_degradation]
                zero_ops_degradation = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_degradtion]

            with tf.device('/cpu:%d' % 0):
                tvs_budget = varlist_budget
                accum_vars_budget =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_budget]
                zero_ops_budget = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_budget]

            with tf.device('/cpu:%d' % 0):
                tvs_utility = varlist_utility + varlist_degradtion
                print(tvs_utility)
                print('###########################################################')
                print(grads_utility)
                accum_vars_utility =  [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs_utility]
                zero_ops_utility = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars_utility]

            #zero_ops = zero_ops_degradation + zero_ops_budget + zero_ops_utility
            global_increment = global_step.assign_add(1)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            # print(tf.get_collection_ref(tf.GraphKeys.MOVING_AVERAGE_VARIABLES))
            #with tf.control_dependencies([tf.group(*update_ops), add_global]):
            with tf.control_dependencies([tf.no_op('update_op')]):
                accum_ops_degradation = [accum_vars_degradtion[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                         enumerate(grads_degradation)]
                accum_ops_utility = [accum_vars_utility[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                     enumerate(grads_utility)]
                accum_ops_budget = [accum_vars_budget[i].assign_add(gv[0] / FLAGS.n_minibatches) for i, gv in
                                    enumerate(grads_budget)]


            #accum_ops = accum_ops_degradation + accum_ops_utility + accum_ops_budget
            with tf.control_dependencies([global_increment]):
                apply_gradient_op_degradation = opt_degradation.apply_gradients([(accum_vars_degradtion[i].value(), gv[1]) for i, gv in enumerate(grads_degradation)], global_step=None)

            apply_gradient_op_utility = opt_utility.apply_gradients([(accum_vars_utility[i].value(), gv[1]) for i, gv in enumerate(grads_utility)], global_step=None)
            apply_gradient_op_budget = opt_budget.apply_gradients([(accum_vars_budget[i].value(), gv[1]) for i, gv in enumerate(grads_budget)], global_step=None)

            train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for f in
                           os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
            val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for f in
                         os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]

            print(train_files)
            print(val_files)

            tr_videos_op, tr_action_labels_op, tr_actor_labels_op = input_data.inputs_videos(filenames = train_files,
                                                 batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                 num_epochs=None,
                                                 num_threads=cfg['DATA']['NUM_THREADS'],
                                                 num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                 shuffle=True)
            val_videos_op, val_action_labels_op, val_actor_labels_op = input_data.inputs_videos(filenames = val_files,
                                                   batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                   num_epochs=None,
                                                   num_threads=cfg['DATA']['NUM_THREADS'],
                                                   num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                   shuffle=True)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            # Create a saver for writing training checkpoints.

            if use_pretrained_model:

                vardict_degradation = {v.name[18:-2]: v for v in varlist_degradtion}
                vardict_utility = {v.name[:-2]: v for v in varlist_utility}
                vardict = dict(vardict_degradation, **vardict_utility)
                restore_model_ckpt(sess, FLAGS.whole_pretraining, vardict, "DegradationModule")
            else:
                saver = tf.train.Saver(tf.trainable_variables())
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            # Create summary writter
            #merged = tf.summary.merge_all()
            #train_writer = tf.summary.FileWriter(FLAGS.log_dir+'train', sess.graph)
            #test_writer = tf.summary.FileWriter(FLAGS.log_dir+'test', sess.graph)

            #gvar_list = tf.global_variables()
            #bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
            #bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
            #saver = tf.train.Saver(tf.global_variables()+tf.local_variables()+bn_moving_vars)
            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=5)
            ckpt_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=20)

            if not os.path.exists(FLAGS.summary_dir):
                os.makedirs(FLAGS.summary_dir)
            loss_summary_file = open(FLAGS.summary_dir+'loss_summary.txt', 'w')
            train_summary_file = open(FLAGS.summary_dir+'train_summary.txt', 'w')
            test_summary_file = open(FLAGS.summary_dir+'test_summary.txt', 'w')
            model_sampling_summary_file = open(FLAGS.summary_dir+'model_summary.txt', 'w')

            largest_gap = 0.15
            actor_uniform_labels = np.full((cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'], cfg['DATA']['NUM_CLASSES_BUDGET']),
                                              1 / cfg['DATA']['NUM_CLASSES_BUDGET'], dtype=np.float32)
            eval_video = np.load('../eval_video.npy')
            eval_video = np.reshape(eval_video, [1, 16, 120, 160, 3])
            for step in xrange(cfg['TRAIN']['MAX_STEPS']):
                if step == 0 or (FLAGS.use_resampling and step % FLAGS.resample_step == 0):
                    budget_varlist = [v for v in tf.trainable_variables() if any(x in v.name for x in ["BudgetModule"])]
                    init_budget_op = tf.variables_initializer(budget_varlist)
                    sess.run(init_budget_op)
                    for _ in itertools.repeat(None, FLAGS.retraining_step):
                        start_time = time.time()
                        acc_util_lst, acc_budget_lst, loss_filter_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        sess.run(zero_ops_budget)
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            tr_videos, tr_action_labels, tr_actor_labels = sess.run(
                                [tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                            _, acc_util, acc_budget, loss_filter_value, loss_utility_value, loss_budget_value = sess.run(
                                [accum_ops_budget, accuracy_util, accuracy_budget, loss_filter_op,
                                 loss_utility_op, loss_budget_op],
                                feed_dict={videos_placeholder: tr_videos,
                                           utility_labels_placeholder: tr_action_labels,
                                           budget_uniform_labels_placeholder: actor_uniform_labels,
                                           budget_labels_placeholder: tr_actor_labels,
                                           dropout_placeholder: 1.0,
                                           isTraining_placeholder: True,
                                           isAdversarial_placeholder:False,
                                           })
                            acc_util_lst.append(acc_util)
                            acc_budget_lst.append(acc_budget)
                            loss_filter_lst.append(loss_filter_value)
                            loss_utility_lst.append(loss_utility_value)
                            loss_budget_lst.append(loss_budget_value)
                        sess.run(apply_gradient_op_budget)
                        assert not np.isnan(np.mean(loss_filter_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Resampling (Budget), Step: {:4d}, time: {:.4f}, budget loss: {:.8f}, training budget accuracy: {:.5f}, ' \
                                       'utility loss: {:.8f}, training utility accuracy: {:.5f}'.format(step,
                                        time.time() - start_time, np.mean(loss_budget_lst), np.mean(acc_budget_lst), np.mean(loss_utility_lst), np.mean(acc_util_lst))
                        model_sampling_summary_file.write(loss_summary + '\n')
                        print(loss_summary)
                start_time = time.time()
                loss_filter_lst, loss_utility_lst, loss_budget_lst = [], [], []
                sess.run(zero_ops_degradation)
                for _ in itertools.repeat(None, FLAGS.n_minibatches):
                    tr_videos, tr_actor_labels, tr_action_labels = sess.run(
                                    [tr_videos_op, tr_actor_labels_op, tr_action_labels_op])
                    _, argmax_cent, loss_budget_uniform_tensor, loss_filter_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_degradation,
                                                        argmax_cent_op, loss_budget_uniform_tensor_op, loss_filter_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={videos_placeholder: tr_videos,
                                                               utility_labels_placeholder: tr_action_labels,
                                                               budget_uniform_labels_placeholder: actor_uniform_labels,
                                                               budget_labels_placeholder: tr_actor_labels,
                                                               dropout_placeholder: 1.0,
                                                               isTraining_placeholder: True,
                                                               isAdversarial_placeholder:True,
                                                               })
                    print(argmax_cent)
                    #print(loss_budget_uniform_tensor)
                    loss_filter_lst.append(loss_filter_value)
                    loss_utility_lst.append(loss_utility_value)
                    loss_budget_lst.append(loss_budget_value)
                _, _lambda = sess.run([apply_gradient_op_degradation, _lambda_op])

                assert not np.isnan(np.mean(loss_filter_lst)), 'Model diverged with loss = NaN'
                duration = time.time() - start_time


                loss_summary = 'Alternating Training (Degradation), Lambda: {:.8f}, Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}'.format(_lambda, step,
                                                        duration, np.mean(loss_filter_lst), np.mean(loss_utility_lst), np.mean(loss_budget_lst))

                print(loss_summary)
                loss_summary_file.write(loss_summary + '\n')

                if FLAGS.use_monitor_utility:
                    while True:
                    #for _ in itertools.repeat(None, FLAGS.adaptation_utility_steps):
                        start_time = time.time()
                        acc_util_lst, acc_budget_lst, loss_value_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            val_videos, val_action_labels, val_actor_labels = sess.run([val_videos_op, val_action_labels_op, val_actor_labels_op])
                            acc_util, acc_budget, loss_value, loss_utility, loss_budget = sess.run(
                                            [accuracy_util, accuracy_budget, loss_op, loss_utility_op, loss_budget_op],
                                                feed_dict={videos_placeholder: val_videos,
                                                    utility_labels_placeholder: val_action_labels,
                                                    budget_uniform_labels_placeholder: actor_uniform_labels,
                                                    budget_labels_placeholder: val_actor_labels,
                                                    dropout_placeholder: 1.0,
                                                    isTraining_placeholder: True,
                                                    isAdversarial_placeholder: True,
                                                })
                            acc_util_lst.append(acc_util)
                            acc_budget_lst.append(acc_budget)
                            loss_value_lst.append(loss_value)
                            loss_utility_lst.append(loss_utility)
                            loss_budget_lst.append(loss_budget)
                            # test_writer.add_summary(summary, step)
                        val_summary = "Step: {:4d}, time: {:.4f}, total loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f},\n" .format(
                                step,
                                time.time() - start_time, np.mean(loss_value_lst),
                                np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                np.mean(acc_util_lst), np.mean(acc_budget_lst))
                        print(val_summary)

                        if np.mean(acc_util_lst) >= FLAGS.highest_util_acc_val:
                            break
                        start_time = time.time()
                        sess.run(zero_ops_utility)
                        acc_util_lst, acc_budget_lst, loss_filter_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                            _, acc_util, acc_budget, loss_filter_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_utility, accuracy_util, accuracy_budget, loss_filter_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={videos_placeholder: tr_videos,
                                                               utility_labels_placeholder: tr_action_labels,
                                                               budget_uniform_labels_placeholder: actor_uniform_labels,
                                                               budget_labels_placeholder: tr_actor_labels,
                                                               dropout_placeholder: 0.5,
                                                               isTraining_placeholder: True,
                                                               isAdversarial_placeholder: True,
                                                               })
                            acc_util_lst.append(acc_util)
                            acc_budget_lst.append(acc_budget)
                            loss_filter_lst.append(loss_filter_value)
                            loss_utility_lst.append(loss_utility_value)
                            loss_budget_lst.append(loss_budget_value)
                        sess.run([apply_gradient_op_utility])
                        assert not np.isnan(np.mean(loss_filter_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Utility), Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, '.format(step,
                                                            time.time() - start_time, np.mean(loss_filter_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst))

                        print(loss_summary)

                if FLAGS.use_monitor_budget:
                    while True:
                        start_time = time.time()
                        sess.run(zero_ops_budget)
                        acc_util_lst, acc_budget_lst, loss_filter_lst, loss_utility_lst, loss_budget_lst = [], [], [], [], []
                        for _ in itertools.repeat(None, FLAGS.n_minibatches):
                            tr_videos, tr_action_labels, tr_actor_labels = sess.run([tr_videos_op, tr_action_labels_op, tr_actor_labels_op])

                            _, acc_util, acc_budget, loss_filter_value, loss_utility_value, loss_budget_value = sess.run([accum_ops_budget, accuracy_util, accuracy_budget, loss_filter_op, loss_utility_op, loss_budget_op],
                                                    feed_dict={videos_placeholder: tr_videos,
                                                               utility_labels_placeholder: tr_action_labels,
                                                               budget_uniform_labels_placeholder: actor_uniform_labels,
                                                               budget_labels_placeholder: tr_actor_labels,
                                                               dropout_placeholder: 1.0,
                                                               isTraining_placeholder: True,
                                                               isAdversarial_placeholder: True,
                                                               })
                            acc_util_lst.append(acc_util)
                            acc_budget_lst.append(acc_budget)
                            loss_filter_lst.append(loss_filter_value)
                            loss_utility_lst.append(loss_utility_value)
                            loss_budget_lst.append(loss_budget_value)
                        sess.run([apply_gradient_op_budget])
                        assert not np.isnan(np.mean(loss_filter_lst)), 'Model diverged with loss = NaN'
                        loss_summary = 'Alternating Training (Budget), Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, ' \
                                   'training utility accuracy: {:.5f}, training budget accuracy: {:.5f}'.format(step,
                                                            time.time() - start_time, np.mean(loss_filter_lst),
                                                            np.mean(loss_utility_lst), np.mean(loss_budget_lst),
                                                            np.mean(acc_util_lst), np.mean(acc_budget_lst))

                        print(loss_summary)
                        if np.mean(acc_budget_lst) >= FLAGS.highest_budget_acc_train:
                            break

                if step % cfg['TRAIN']['VAL_STEP'] == 0:
                    eval_video_output = np.squeeze(
                        sess.run(eval_video_op, feed_dict={eval_video_placeholder: eval_video}))
                    print(eval_video_output.shape)
                    plot_visualization_frame(eval_video_output[6], step)
                    start_time = time.time()
                    acc_util_train_lst, acc_budget_train_lst, loss_filter_train_lst, loss_utility_train_lst, loss_budget_train_lst = [], [], [], [], []
                    acc_util_val_lst, acc_budget_val_lst, loss_filter_val_lst, loss_utility_val_lst, loss_budget_val_lst = [], [], [], [], []
                    for _ in itertools.repeat(None, FLAGS.n_minibatches):
                        tr_videos, tr_action_labels, tr_actor_labels = sess.run(
                                    [tr_videos_op, tr_action_labels_op, tr_actor_labels_op])
                        acc_util, acc_budget, loss_filter_value, loss_utility_value, loss_budget_value = sess.run([accuracy_util, accuracy_budget,
                                                                                                loss_filter_op, loss_utility_op, loss_budget_op],
                                                           feed_dict={videos_placeholder: tr_videos,
                                                                      utility_labels_placeholder: tr_action_labels,
                                                                      budget_uniform_labels_placeholder: actor_uniform_labels,
                                                                      budget_labels_placeholder: tr_actor_labels,
                                                                      dropout_placeholder: 1.0,
                                                                      isTraining_placeholder: True,
                                                                      isAdversarial_placeholder: True,
                                                                      })
                        acc_util_train_lst.append(acc_util)
                        acc_budget_train_lst.append(acc_budget)
                        loss_filter_train_lst.append(loss_filter_value)
                        loss_utility_train_lst.append(loss_utility_value)
                        loss_budget_train_lst.append(loss_budget_value)

                    train_summary = "Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, training utility accuracy: {:.5f}, training budget accuracy: {:.5f}".format(
                            step,
                            time.time() - start_time, np.mean(loss_filter_train_lst),
                            np.mean(loss_utility_train_lst), np.mean(loss_budget_train_lst),
                            np.mean(acc_util_train_lst), np.mean(acc_budget_train_lst))
                    print(train_summary)
                    train_summary_file.write(train_summary + '\n')

                    for _ in itertools.repeat(None, FLAGS.n_minibatches_eval):
                        val_videos, val_action_labels, val_actor_labels = sess.run(
                                        [val_videos_op, val_action_labels_op, val_actor_labels_op])
                        acc_util, acc_budget, loss_filter_value, loss_utility_value, loss_budget_value = sess.run([accuracy_util, accuracy_budget,
                                                                                                loss_filter_op, loss_utility_op, loss_budget_op],
                                                            feed_dict={videos_placeholder: val_videos,
                                                                        utility_labels_placeholder: val_action_labels,
                                                                        budget_uniform_labels_placeholder: actor_uniform_labels,
                                                                        budget_labels_placeholder: val_actor_labels,
                                                                        dropout_placeholder: 1.0,
                                                                        isTraining_placeholder: True,
                                                                        isAdversarial_placeholder: True,
                                                                       })
                        acc_util_val_lst.append(acc_util)
                        acc_budget_val_lst.append(acc_budget)
                        loss_filter_val_lst.append(loss_filter_value)
                        loss_utility_val_lst.append(loss_utility_value)
                        loss_budget_val_lst.append(loss_budget_value)

                    test_summary = "Step: {:4d}, time: {:.4f}, filter loss: {:.8f}, utility loss: {:.8f}, budget loss: {:.8f}, validation utility accuracy: {:.5f}, validation budget accuracy: {:.5f}".format(step,
                                                            time.time() - start_time, np.mean(loss_filter_val_lst),
                                                            np.mean(loss_utility_val_lst), np.mean(loss_budget_val_lst),
                                                            np.mean(acc_util_val_lst), np.mean(acc_budget_val_lst))
                    print(test_summary)
                    test_summary_file.write(test_summary + '\n')

                gap = np.mean(acc_util_val_lst) - np.mean(acc_budget_val_lst)

                if gap > largest_gap and step % FLAGS.resample_step != 0:
                    largest_gap = gap
                    checkpoint_path = os.path.join(FLAGS.saved_checkpoint_dir, '{}_model.ckpt'.format(gap))
                    ckpt_saver.save(sess, checkpoint_path, global_step=step)

                if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == cfg['TRAIN']['MAX_STEPS']:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            loss_summary_file.close()
            train_summary_file.close()
            test_summary_file.close()
            coord.request_stop()
            coord.join(threads)

    print("done")

# Run testing of the trained model
# It will give the utility task accuracy and the privacy budget task accuracy
def run_testing_model_sampling(cfg):
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    multiplier_lst = [0.54 - i * 0.02 for i in range(FLAGS.NBudget)]
    from collections import defaultdict
    logits_budget_lst_dct = defaultdict(list)

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            videos_placeholder, utility_labels_placeholder, budget_labels_placeholder, dropout_placeholder, _ = placeholder_inputs(cfg, cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'])
            isTraining_placeholder = tf.placeholder(tf.bool)
            isAdversarial_placeholder = tf.placeholder(tf.bool)

            # Compute Acc
            logits_utility_lst = []
            logits_budget_lst = []

            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            LR_videos = residualNet(videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], is_video=True)
                            if cfg['DATA_PROCESSING']['USE_INSTANCE_NORM']:
                                LR_videos = _instance_norm(LR_videos)
                            if cfg['DATA_PROCESSING']['USE_BILINEAR_RESIZE']:
                                LR_videos = _bilinear_resize(LR_videos)
                            if cfg['DATA_PROCESSING']['USE_BINARIZE']:
                                LR_videos = tf.cond(isAdversarial_placeholder, lambda: tf.identity(LR_videos), lambda: _binary_activation(LR_videos))

                            logits_utility = utilityNet(LR_videos, dropout_placeholder, wd=0.005)
                            logits_utility_lst.append(logits_utility)
                            logits_budget = tf.zeros([cfg['TRAIN']['BATCH_SIZE'], cfg['DATA']['NUM_CLASSES_BUDGET']])
                            for multiplier in multiplier_lst:
                                logits = budgetNet(LR_videos, isTraining_placeholder, depth_multiplier=multiplier)
                                print(multiplier)
                                print('##################################################################')
                                logits_budget_lst_dct['{}'.format(multiplier)].append(logits)
                                logits_budget += logits
                            logits_budget_lst.append(logits_budget)

                            tf.get_variable_scope().reuse_variables()

            logits_utility = tf.concat(logits_utility_lst, 0)
            logits_budget = tf.concat(logits_budget_lst, 0)
            right_count_utility_op = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_utility), axis=1), utility_labels_placeholder), tf.int32))
            #softmax_logits_utility_op = tf.nn.softmax(logits_utility)

            right_count_budget_op =  tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_budget), axis=1), budget_labels_placeholder), tf.int32))
            #softmax_logits_budget_op = tf.nn.softmax(logits_budget)

            right_count_budget_op_lst = []
            for multiplier in multiplier_lst:
                logits = tf.concat(logits_budget_lst_dct['{}'.format(multiplier)], 0)
                right_count_op = tf.reduce_sum(
                    tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), budget_labels_placeholder),
                            tf.int32))
                right_count_budget_op_lst.append(right_count_op)

            train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for f in
                           os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
            val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for f in
                         os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]
            test_files = [os.path.join(cfg['DATA']['TEST_FILES_DIR'], f) for f in
                          os.listdir(cfg['DATA']['TEST_FILES_DIR']) if f.endswith('.tfrecords')]

            print(train_files)
            print(val_files)
            print(test_files)

            videos_op, action_labels_op, actor_labels_op = input_data.inputs_videos(filenames = val_files + test_files,
                                                   batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                   num_epochs=1,
                                                   num_threads=cfg['DATA']['NUM_THREADS'],
                                                   num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                   shuffle=False)



            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Create a saver for writing training checkpoints.
            saver = tf.train.Saver(tf.trainable_variables())
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)
            total_v_utility = 0.0
            total_v_budget = 0.0

            test_correct_num_utility = 0.0
            test_correct_num_budget = 0.0
            test_correct_num_budget_lst = [0.0] * FLAGS.NBudget

            try:
                while not coord.should_stop():
                    videos, utility_labels, budget_labels = sess.run([videos_op, action_labels_op, actor_labels_op])
                    feed = {videos_placeholder: videos, budget_labels_placeholder: budget_labels,
                            utility_labels_placeholder: utility_labels, isTraining_placeholder: True,
                            dropout_placeholder: 1.0, isAdversarial_placeholder: True}
                    right_counts  = sess.run([right_count_utility_op, right_count_budget_op] + right_count_budget_op_lst, feed_dict=feed)

                    test_correct_num_utility += right_counts[0]
                    total_v_utility += utility_labels.shape[0]

                    test_correct_num_budget += right_counts[1]
                    total_v_budget += budget_labels.shape[0]

                    for i in range(FLAGS.NBudget):
                        test_correct_num_budget_lst[i] += right_counts[i+2]
                    # print(tf.argmax(softmax_logits, 1).eval(session=sess))
                    # print(logits.eval(feed_dict=feed, session=sess))
                    # print(labels)
            except tf.errors.OutOfRangeError:
                print('Done testing on all the examples')
            finally:
                coord.request_stop()

            with open('EvaluationResuls.txt', 'w') as wf:
                wf.write('Utility test acc: {},\ttest_correct_num: {},\ttotal_v: {}\n'.format(
                    test_correct_num_utility / total_v_utility, test_correct_num_utility, total_v_utility))
                wf.write('Budget ensemble test acc: {},\ttest_correct_num: {},\ttotal_v: {}\n'.format(
                    test_correct_num_budget / total_v_budget, test_correct_num_budget, total_v_budget))

                for i in range(FLAGS.NBudget):
                    wf.write('Budget{} test acc: {},\ttest_correct_num: {}\t: total_v: {}\n'.format(
                        multiplier_lst[i], test_correct_num_budget_lst[i] / total_v_budget, test_correct_num_budget_lst[i], total_v_budget))

            coord.join(threads)
            sess.close()

def main(_):
    print(os.getcwd())
    print(sys.path)
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
    cfg = yaml.load(open('params.yml'))
    pp.pprint(cfg)
    run_training_model_sampling(cfg)
    #run_training_ensembling_restarting_nentropy(cfg)

if __name__ == '__main__':
    tf.app.run()
