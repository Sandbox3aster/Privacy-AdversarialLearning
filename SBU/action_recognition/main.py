import os
import sys
sys.path.insert(0, '..')

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,7"

import tensorflow as tf
from img_proc import _bilinear_resize
from loss import  *
from utilityNet import utilityNet
from input_data import *
import errno
import time
from utils import *
from six.moves import xrange
import yaml
import pprint

def run_training_utility(cfg, factor):
    # Create model directory
    if not os.path.exists(os.path.join(cfg['MODEL']['UTILITY_MODEL'], str(factor))):
        os.makedirs(os.path.join(cfg['MODEL']['UTILITY_MODEL'], str(factor)))

    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder = tf.placeholder(tf.float32, shape=(cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'], cfg['DATA']['DEPTH'], 112, 112, cfg['DATA']['NCHANNEL']))
            utility_labels_placeholder = tf.placeholder(tf.int64, shape=(cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM']))
            dropout_placeholder = tf.placeholder(tf.float32)

            tower_grads_utility = []
            logits_utility = []
            losses_utility = []
            opt_utility = tf.train.AdamOptimizer(1e-4)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            X = _bilinear_resize(videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], factor)
                            logit_utility = utilityNet(X, dropout_placeholder)
                            logits_utility.append(logit_utility)
                            loss_utility = tower_loss_xentropy_sparse(
                                scope,
                                logit_utility,
                                utility_labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
                                )
                            losses_utility.append(loss_utility)
                            varlist_utility = tf.trainable_variables()

                            #varlist_utility = [v for v in tf.trainable_variables() if any(x in v.name for x in ["out", "d2"])]
                            print([v.name for v in varlist_utility])

                            grads_utility = opt_utility.compute_gradients(loss_utility, varlist_utility)
                            tower_grads_utility.append(grads_utility)

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
            loss_op = tf.reduce_mean(losses_utility, name='softmax')
            logits_utility = tf.concat(logits_utility, 0)
            accuracy_util = accuracy(logits_utility, utility_labels_placeholder)

            grads_utility = average_gradients(tower_grads_utility)
            apply_gradient_op_utility = opt_utility.apply_gradients(grads_utility, global_step=global_step)

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(cfg['TRAIN']['MOVING_AVERAGE_DECAY'], global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_op = tf.group(apply_gradient_op_utility, variables_averages_op)

            train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for
                           f in os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
            val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for
                         f in os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]
            test_files = [os.path.join(cfg['DATA']['TEST_FILES_DIR'], f) for
                          f in os.listdir(cfg['DATA']['TEST_FILES_DIR']) if f.endswith('.tfrecords')]

            print(train_files)
            print(val_files)

            tr_videos_op, tr_utility_labels_op, _ = inputs_videos(filenames = train_files,
                                                 batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                 num_epochs=None,
                                                 num_threads=cfg['DATA']['NUM_THREADS'],
                                                 num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                 shuffle=True)
            val_videos_op, val_utility_labels_op, _ = inputs_videos(filenames = val_files,
                                                   batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                   num_epochs=None,
                                                   num_threads=cfg['DATA']['NUM_THREADS'],
                                                   num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                   shuffle=True)
            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Create a saver for writing training checkpoints.
            if use_pretrained_model:
                if os.path.isfile(cfg['MODEL']['PRETRAINED_C3D']):
                    varlist = [v for v in tf.trainable_variables() if not any(x in v.name.split('/')[1] for x in ["out", "d2"])]
                    vardict = {v.name[:-2].replace('UtilityModule', 'var_name'):v for v in varlist}
                    for key, value in vardict.items():
                        print(key)
                    saver = tf.train.Saver(vardict)
                    saver.restore(sess, cfg['MODEL']['PRETRAINED_C3D'])
                    print('#############################Session restored from pretrained model at {}!#############################'.format(cfg['MODEL']['PRETRAINED_C3D']))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cfg['MODEL']['PRETRAINED_C3D'])
            else:
                saver = tf.train.Saver(tf.trainable_variables())
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=cfg['MODEL']['UTILITY_MODEL'])
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('#############################Session restored from trained model at {}!#############################'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cfg['MODEL']['UTILITY_MODEL'])
            # Create summary writter
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(cfg['DATA']['LOG_DIR']+'train', sess.graph)
            test_writer = tf.summary.FileWriter(cfg['DATA']['LOG_DIR']+'test', sess.graph)
            saver = tf.train.Saver()
            for step in xrange(cfg['TRAIN']['MAX_STEPS']):
                start_time = time.time()
                tr_videos, tr_utility_labels = sess.run(
                    [tr_videos_op, tr_utility_labels_op])
                assert not np.any(np.isnan(tr_videos)), 'Video data has NAN value'
                print(tr_videos.shape)
                print(tr_utility_labels.shape)
                _, loss_value = sess.run([train_op, loss_op], feed_dict={videos_placeholder: tr_videos,
                    utility_labels_placeholder: tr_utility_labels, dropout_placeholder: 0.5})
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                print('Step: {:4d} time: {:.4f} loss: {:.8f}'.format(step, time.time() - start_time, loss_value))
                if step % cfg['TRAIN']['VAL_STEP'] == 0:
                    start_time = time.time()
                    tr_videos, tr_utility_labels= sess.run(
                        [tr_videos_op, tr_utility_labels_op])
                    summary, acc_util, loss_value = sess.run([merged, accuracy_util, loss_op],
                                                           feed_dict={videos_placeholder: tr_videos,
                                                                      utility_labels_placeholder: tr_utility_labels,
                                                                      dropout_placeholder: 1.0})
                    print("Step: {:4d} time: {:.4f}, training utility accuracy: {:.5f}, loss: {:.8f}".
                        format(step, time.time()-start_time, acc_util, loss_value))

                    train_writer.add_summary(summary, step)

                    start_time = time.time()
                    val_videos, val_utility_labels = sess.run(
                        [val_videos_op, val_utility_labels_op])
                    summary, acc_util, loss_value = sess.run(
                        [merged, accuracy_util, loss_op],
                        feed_dict={videos_placeholder: val_videos,
                                   utility_labels_placeholder: val_utility_labels,
                                   dropout_placeholder: 1.0})
                    print("Step: {:4d} time: {:.4f}, validation utility accuracy: {:.5f}, loss: {:.8f}".
                        format(step, time.time() - start_time, acc_util, loss_value))
                    test_writer.add_summary(summary, step)
                # Save a checkpoint and evaluate the model periodically.
                if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == cfg['TRAIN']['MAX_STEPS']:
                    checkpoint_path = os.path.join(os.path.join(cfg['MODEL']['UTILITY_MODEL'], str(factor)), 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

    print("done")

def run_testing_utility(cfg, factor):
    tf.reset_default_graph()
    videos_placeholder = tf.placeholder(tf.float32, shape=(cfg['TEST']['BATCH_SIZE'] * cfg['TEST']['GPU_NUM'], cfg['DATA']['DEPTH'], 112, 112, cfg['DATA']['NCHANNEL']))
    utility_labels_placeholder = tf.placeholder(tf.int64, shape=(cfg['TEST']['BATCH_SIZE']  * cfg['TEST']['GPU_NUM']))
    dropout_placeholder = tf.placeholder(tf.float32)

    logits = []

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(0, cfg['TEST']['GPU_NUM']):
            with tf.device('/gpu:%d' % gpu_index):
                print('/gpu:%d' % gpu_index)
                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                    X = _bilinear_resize(videos_placeholder[gpu_index * cfg['TEST']['BATCH_SIZE'] :(gpu_index + 1) * cfg['TEST']['BATCH_SIZE'] ], factor=factor)
                    logit = utilityNet(X, dropout_placeholder)
                    logits.append(logit)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()


    logits = tf.concat(logits, 0)
    right_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), utility_labels_placeholder), tf.int32))
    softmax_logits_op = tf.nn.softmax(logits)

    train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for f in
                   os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
    val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for f in
                 os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]
    test_files = [os.path.join(cfg['DATA']['TEST_FILES_DIR'], f) for f in
                  os.listdir(cfg['DATA']['TEST_FILES_DIR']) if f.endswith('.tfrecords')]

    print(train_files)
    print(val_files)
    videos_op, labels_op, _ = inputs_videos(filenames=val_files+test_files,
                                       batch_size=cfg['TEST']['BATCH_SIZE'] * cfg['TEST']['GPU_NUM'],
                                       num_epochs=1,
                                       num_threads=cfg['DATA']['NUM_THREADS'],
                                       num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                       shuffle=False
                                       )

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver(tf.trainable_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(cfg['MODEL']['UTILITY_MODEL'], str(factor)))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(
            '#############################Session restored from trained model at {}!#############################'.format(
                ckpt.model_checkpoint_path))
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(cfg['MODEL']['UTILITY_MODEL'], str(factor)))
    total_v = 0.0
    test_correct_num = 0.0
    try:
        while not coord.should_stop():
            videos, labels = sess.run([videos_op, labels_op])
            feed = {videos_placeholder: videos, utility_labels_placeholder: labels, dropout_placeholder: 1.0}
            right, softmax_logits= sess.run([right_count, softmax_logits_op], feed_dict=feed)
            test_correct_num += right
            total_v += labels.shape[0]
            print(softmax_logits.shape)
            print(sess.run(tf.argmax(softmax_logits, 1)))
    except tf.errors.OutOfRangeError:
        print('Done testing on all the examples')
    finally:
        coord.request_stop()
    print('test acc:', test_correct_num / total_v, 'test_correct_num:', test_correct_num,
              'total_v:', total_v)
    with open('utility_evaluation_{}.txt'.format(factor), 'w') as wf:
        wf.write('test acc: {}\ttest_correct_num:{}\ttotal_v\n'.format(
            test_correct_num / total_v, test_correct_num, total_v))
    coord.join(threads)
    sess.close()

def main(_):
    cfg = yaml.load(open('params.yml'))
    pp = pprint.PrettyPrinter()
    pp.pprint(cfg)

    # Different down-sample factor
    for factor in [2, 4, 6, 8, 14, 16, 28, 56]:
        run_training_utility(cfg, factor)
        run_testing_utility(cfg, factor)


if __name__ == '__main__':
    tf.app.run()