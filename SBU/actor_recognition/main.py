import time

import os
import tensorflow as tf

import sys
sys.path.insert(0, '..')

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="0,4"

from img_proc import _bilinear_resize
from loss import  *
from utils import *
from budgetNet import budgetNet
from input_data import *
import errno
from six.moves import xrange

import yaml
import pprint

def run_training_budget(cfg, factor):
    # Create model directory
    if not os.path.exists(os.path.join(cfg['MODEL']['BUDGET_MODEL'], str(factor))):
        os.makedirs(os.path.join(cfg['MODEL']['BUDGET_MODEL'], str(factor)))

    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder = tf.placeholder(tf.float32, shape=(cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'], cfg['DATA']['DEPTH'], 112, 112, cfg['DATA']['NCHANNEL']))
            labels_placeholder = tf.placeholder(tf.int64, shape=(cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM']))

            isTraining_placeholder = tf.placeholder(tf.bool)
            tower_grads_finetune = []
            tower_grads_major = []
            logits = []
            losses = []
            opt_finetune = tf.train.AdamOptimizer(1e-3)
            opt_major = tf.train.AdamOptimizer(1e-4)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, cfg['TRAIN']['GPU_NUM']):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            X = _bilinear_resize(
                                videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']], factor)
                            logit = budgetNet(X, isTraining_placeholder)
                            logits.append(logit)
                            loss = tower_loss_xentropy_sparse(
                                scope,
                                logit,
                                labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
                                )
                            losses.append(loss)

                            varlist_finetune = [v for v in tf.trainable_variables() if any(x in v.name for x in ["Conv2d_1c_1x1"])]
                            #varlist = tf.trainable_variables()
                            print([v.name for v in varlist_finetune])
                            varlist_major = [v for v in tf.trainable_variables() if not any(x in v.name for x in ["Conv2d_1c_1x1"])]
                            print([v.name for v in varlist_major])
                            grads_finetune = opt_finetune.compute_gradients(loss, varlist_finetune)
                            grads_major = opt_major.compute_gradients(loss, varlist_major)
                            tower_grads_finetune.append(grads_finetune)
                            tower_grads_major.append(grads_major)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
            loss_op = tf.reduce_mean(losses, name='softmax')
            logits = tf.concat(logits, 0)
            acc_op = accuracy(logits, labels_placeholder)
            tf.summary.scalar('budget task accuracy', acc_op)
            grads_finetune = average_gradients(tower_grads_finetune)
            grads_major = average_gradients(tower_grads_major)

            apply_gradient_op_budget_finetune = opt_finetune.apply_gradients(grads_finetune, global_step=global_step)
            apply_gradient_op_budget_major = opt_major.apply_gradients(grads_major, global_step=global_step)
            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(cfg['TRAIN']['MOVING_AVERAGE_DECAY'], global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print(update_ops)
            #print(tf.get_collection_ref(tf.GraphKeys.MOVING_AVERAGE_VARIABLES))
            with tf.control_dependencies([tf.group(*update_ops)]):
                train_op = tf.group(apply_gradient_op_budget_finetune, apply_gradient_op_budget_major, variables_averages_op)

            train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for
                           f in os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
            val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for
                         f in os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]
            test_files = [os.path.join(cfg['DATA']['TEST_FILES_DIR'], f) for
                          f in os.listdir(cfg['DATA']['TEST_FILES_DIR']) if f.endswith('.tfrecords')]

            print(train_files)
            print(val_files)

            tr_videos_op, _, tr_budget_labels_op = inputs_videos(filenames = train_files,
                                                 batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
                                                 num_epochs=None,
                                                 num_threads=cfg['DATA']['NUM_THREADS'],
                                                 num_examples_per_epoch=cfg['TRAIN']['NUM_EXAMPLES_PER_EPOCH'],
                                                 shuffle=True)
            val_videos_op, _, val_budget_labels_op = inputs_videos(filenames = val_files,
                                                   batch_size=cfg['TRAIN']['BATCH_SIZE'] * cfg['TRAIN']['GPU_NUM'],
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
            # Create a saver for writing training checkpoints.
            if use_pretrained_model:
                varlist = [v for v in tf.trainable_variables() if not any(x in v.name for x in ["Conv2d_1c_1x1"])]
                varlist += bn_moving_vars
                vardict = {v.name[:-2].replace('BudgetModule_{}'.format(1.0), 'MobilenetV1'): v for v in varlist}
                saver = tf.train.Saver(vardict)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=cfg['MODEL']['PRETRAINED_MOBILENET_10'])
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('#############################Session restored from pretrained model at {}!###############################'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cfg['MODEL']['PRETRAINED_MOBILENET_10'])
            else:
                varlist = tf.trainable_variables()
                varlist += bn_moving_vars
                saver = tf.train.Saver(varlist)
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(cfg['MODEL']['BUDGET_MODEL'], format(factor)))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('#############################Session restored from trained model at {}!###############################'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(cfg['MODEL']['BUDGET_MODEL'], format(factor)))

            # Create summary writter
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(cfg['DATA']['LOG_DIR']+'train', sess.graph)
            test_writer = tf.summary.FileWriter(cfg['DATA']['LOG_DIR']+'test', sess.graph)
            varlist = tf.trainable_variables()
            varlist += bn_moving_vars
            saver = tf.train.Saver(varlist, max_to_keep=20)
            for step in xrange(1000):
                start_time = time.time()
                tr_videos, tr_labels = sess.run(
                    [tr_videos_op, tr_budget_labels_op])
                _, loss_value = sess.run([train_op, loss_op], feed_dict={videos_placeholder: tr_videos,
                                                                         labels_placeholder: tr_labels,
                                                                         isTraining_placeholder: True})
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                duration = time.time() - start_time
                print('Step: {:4d} time: {:.4f} loss: {:.8f}'.format(step, duration, loss_value))
                if step % cfg['TRAIN']['VAL_STEP'] == 0:
                    start_time = time.time()
                    tr_videos, tr_labels= sess.run(
                            [tr_videos_op, tr_budget_labels_op])
                    summary, acc, loss_value = sess.run([merged, acc_op, loss_op],
                                                           feed_dict={videos_placeholder: tr_videos,
                                                                      labels_placeholder: tr_labels,
                                                                      isTraining_placeholder: False})
                    print("Step: {:4d} time: {:.4f}, training budget accuracy: {:.5f}, loss: {:.8f}".
                          format(step, time.time()-start_time, acc, loss_value))

                    train_writer.add_summary(summary, step)

                    start_time = time.time()
                    val_videos, val_labels = sess.run(
                            [val_videos_op, val_budget_labels_op])
                    summary, acc, loss_value = sess.run([merged, acc_op, loss_op],
                                                            feed_dict={videos_placeholder: val_videos,
                                                                       labels_placeholder: val_labels,
                                                                       isTraining_placeholder: False})
                    print("Step: {:4d} time: {:.4f}, validation budget accuracy: {:.5f}, loss: {:.8f}".
                        format(step, time.time() - start_time, acc, loss_value))
                    test_writer.add_summary(summary, step)
                # Save a checkpoint and evaluate the model periodically.
                if step % cfg['TRAIN']['SAVE_STEP'] == 0 or (step + 1) == cfg['TRAIN']['MAX_STEPS']:
                    checkpoint_path = os.path.join(os.path.join(cfg['MODEL']['BUDGET_MODEL'], str(factor)), 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

    print("done")

def run_testing_budget(cfg, factor):
    tf.reset_default_graph()

    videos_placeholder = tf.placeholder(tf.float32, shape=(cfg['TEST']['BATCH_SIZE'] * cfg['TEST']['GPU_NUM'], cfg['DATA']['DEPTH'], 112, 112, cfg['DATA']['NCHANNEL']))
    labels_placeholder = tf.placeholder(tf.int64, shape=(cfg['TEST']['BATCH_SIZE'] * cfg['TEST']['GPU_NUM']))

    isTraining_placeholder = tf.placeholder(tf.bool)
    logits = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(0,cfg['TEST']['GPU_NUM']):
            with tf.device('/gpu:%d' % gpu_index):
                print('/gpu:%d' % gpu_index)
                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                    X = _bilinear_resize(videos_placeholder[gpu_index * cfg['TEST']['BATCH_SIZE']:(gpu_index + 1) * cfg['TEST']['BATCH_SIZE']], factor=factor)
                    logit = budgetNet(X, isTraining_placeholder)
                    logits.append(logit)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
    logits = tf.concat(logits, 0)

    right_count_op = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), labels_placeholder), tf.int32))
    softmax_logits_op = tf.nn.softmax(logits)

    train_files = [os.path.join(cfg['DATA']['TRAIN_FILES_DIR'], f) for
                   f in os.listdir(cfg['DATA']['TRAIN_FILES_DIR']) if f.endswith('.tfrecords')]
    val_files = [os.path.join(cfg['DATA']['VAL_FILES_DIR'], f) for
                 f in os.listdir(cfg['DATA']['VAL_FILES_DIR']) if f.endswith('.tfrecords')]
    test_files = [os.path.join(cfg['DATA']['TEST_FILES_DIR'], f) for
                  f in os.listdir(cfg['DATA']['TEST_FILES_DIR']) if f.endswith('.tfrecords')]

    print(train_files)
    print(val_files)

    videos_op, _, labels_op = inputs_videos(filenames=train_files,
                                                             batch_size=cfg['TEST']['BATCH_SIZE'] *cfg['TEST']['GPU_NUM'],
                                                             num_epochs=1,
                                                             num_threads=cfg['DATA']['NUM_THREADS'],
                                                             num_examples_per_epoch=cfg['TEST']['NUM_EXAMPLES_PER_EPOCH'],
                                                             shuffle=False)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tvars = tf.trainable_variables()
    print('----------------------------Trainable Variables-----------------------------------------')
    gvar_list = tf.global_variables()
    bn_moving_vars = [g for g in gvar_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in gvar_list if 'moving_variance' in g.name]
    saver = tf.train.Saver(tf.trainable_variables()+bn_moving_vars)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(cfg['MODEL']['BUDGET_MODEL'], str(factor)))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Session restored from pretrained budget model at {}!'.format(ckpt.model_checkpoint_path))
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), os.path.join(cfg['MODEL']['BUDGET_MODEL'], str(factor)))
    total_v = 0.0
    test_correct_num = 0.0
    try:
        while not coord.should_stop():
            videos, labels = sess.run([videos_op, labels_op])
            #write_video(videos, labels)
            feed = {videos_placeholder: videos, labels_placeholder: labels, isTraining_placeholder: False}
            right, softmax_logits = sess.run([right_count_op, softmax_logits_op], feed_dict=feed)
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

    with open('budget_evaluation_training_{}.txt'.format(factor), 'w') as wf:
        wf.write('test acc:{}\ttest_correct_num:{}\ttotal_v:{}\t'.format(test_correct_num / total_v, test_correct_num, total_v))

    coord.join(threads)
    sess.close()

def main(_):
    cfg = yaml.load(open('params.yml'))
    pp = pprint.PrettyPrinter()
    pp.pprint(cfg)

    # Different down-sample factor
    for factor in [2, 4, 6, 8, 14, 16, 28, 56]:
        run_training_budget(cfg, factor)
        run_testing_budget(cfg, factor)


if __name__ == '__main__':
    tf.app.run()