#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import time
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
from tf_flags import FLAGS
import errno
import pprint


def placeholder_inputs():
    videos_placeholder = tf.placeholder(tf.float32, shape=(None, FLAGS.depth, FLAGS.height, FLAGS.width, FLAGS.nchannel))
    labels_placeholder = tf.placeholder(tf.int64, shape=(None))
    dropout_placeholder = tf.placeholder(tf.float32)
    return videos_placeholder, labels_placeholder, dropout_placeholder

def tower_loss_xentropy(name_scope, logit, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))
    tf.summary.scalar(name_scope + 'cross entropy', cross_entropy_mean)

    weight_decay_loss = tf.add_n(tf.get_collection('losses', name_scope))
    tf.summary.scalar(name_scope + 'weight decay loss', weight_decay_loss)

    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.get_collection('losses', name_scope)

    # Calculate the total loss for the current tower.
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    total_loss = tf.add_n(losses, name='total_loss')
    tf.summary.scalar(name_scope + 'total loss', total_loss)

    return total_loss

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    use_pretrained_model = True

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            videos_placeholder, labels_placeholder, dropout_placeholder = placeholder_inputs()
            tower_grads1 = []
            tower_grads2 = []
            logits = []
            losses = []
            opt_stable = tf.train.AdamOptimizer(1e-4)
            opt_finetuning = tf.train.AdamOptimizer(1e-5)
            with tf.variable_scope(tf.get_variable_scope()) as scope:
                for gpu_index in range(0, FLAGS.gpu_num):
                    with tf.device('/gpu:%d' % gpu_index):
                        print('/gpu:%d' % gpu_index)
                        with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                            logit = c3d_model.inference_c3d(
                                videos_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                                dropout_placeholder,
                                FLAGS.batch_size
                                )
                            loss = tower_loss_xentropy(
                                scope,
                                logit,
                                labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                                )
                            losses.append(loss)
                            varlist1 = [v for v in tf.trainable_variables() if
                                            not any(x in v.name for x in ["out", "d2"])]
                            varlist2 = [v for v in tf.trainable_variables() if any(x in v.name for x in ["out", "d2"])]

                            print('######################varlist1######################')
                            print([v.name for v in varlist1])
                            print('######################varlist2######################')
                            print([v.name for v in varlist2])
                            #grads1 = opt_stable.compute_gradients(loss, varlist1)
                            grads2 = opt_finetuning.compute_gradients(loss, varlist2)
                            #tower_grads1.append(grads1)
                            tower_grads2.append(grads2)
                            logits.append(logit)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
            logits = tf.concat(logits, 0)
            loss_op = tf.reduce_mean(losses, name='softmax')
            accuracy = c3d_model.accuracy(logits, labels_placeholder)
            tf.summary.scalar('accuracy', accuracy)

            #grads1 = average_gradients(tower_grads1)
            grads2 = average_gradients(tower_grads2)

            #apply_gradient_op1 = opt_stable.apply_gradients(grads1, global_step=global_step)
            apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
            #train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
            train_op = tf.group(apply_gradient_op2)

            train_files = [os.path.join(FLAGS.training_data, f) for f in
                           os.listdir(FLAGS.training_data) if f.endswith('.tfrecords')]
            val_files = [os.path.join(FLAGS.validation_data, f) for f in
                           os.listdir(FLAGS.validation_data) if f.endswith('.tfrecords')]
            print(train_files)
            print(val_files)
            tr_videos_op, tr_labels_op = input_data.inputs(filenames = train_files,
                                                 batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                 num_epochs=None,
                                                 num_threads=FLAGS.num_threads,
                                                 num_examples_per_epoch=FLAGS.num_examples_per_epoch)
            val_videos_op, val_labels_op = input_data.inputs(filenames = val_files,
                                                   batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                                   num_epochs=None,
                                                   num_threads=FLAGS.num_threads,
                                                   num_examples_per_epoch=FLAGS.num_examples_per_epoch)
            init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # Model restoring.
            if use_pretrained_model:
                if os.path.isfile(FLAGS.pretrained_model):
                    varlist = [v for v in tf.trainable_variables() if
                                            not any(x in v.name.split('/')[1] for x in ["out", "d2"])]
                    vardict = {v.name[:-2].replace('C3DNet', 'var_name'): v for v in varlist}
                    for key, value in vardict.items():
                        print(key)
                    saver = tf.train.Saver(vardict)
                    saver.restore(sess, FLAGS.pretrained_model)
                    print('Session restored from pretrained model at {}!'.format(FLAGS.pretrained_model))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.pretrained_model)
            else:
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Session restored from trained model at {}!'.format(ckpt.model_checkpoint_path))
                else:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.checkpoint_dir)

            # Create summary writter
            merge_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(FLAGS.log_dir+'train', sess.graph)
            test_writer = tf.summary.FileWriter(FLAGS.log_dir+'test', sess.graph)
            saver = tf.train.Saver(tf.trainable_variables())
            for step in range(FLAGS.max_steps):
                start_time = time.time()
                tr_videos, tr_labels = sess.run([tr_videos_op, tr_labels_op])
                _, loss_value = sess.run([train_op, loss_op], feed_dict={videos_placeholder: tr_videos, labels_placeholder: tr_labels, dropout_placeholder: 0.5})
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                duration = time.time() - start_time
                print('Step: {:4d} time: {:.4f} loss: {:.8f}'.format(step, duration, loss_value))
                if step % FLAGS.val_step == 0:
                    start_time = time.time()
                    tr_videos, tr_labels = sess.run([tr_videos_op, tr_labels_op])
                    summary, acc, loss_value = sess.run([merge_op, accuracy, loss_op],
                                                           feed_dict={videos_placeholder: tr_videos,
                                                                      labels_placeholder: tr_labels,
                                                                      dropout_placeholder: 1.0})
                    print("Step: {:4d} time: {:.4f}, training accuracy: {:.5f}, loss: {:.8f}".format(step, time.time()-start_time, acc, loss_value))
                    train_writer.add_summary(summary, step)

                    start_time = time.time()
                    val_videos, val_labels = sess.run([val_videos_op, val_labels_op])
                    summary, acc, loss_value = sess.run([merge_op, accuracy, loss_op],
                                                           feed_dict={videos_placeholder: val_videos,
                                                                      labels_placeholder: val_labels,
                                                                      dropout_placeholder: 1.0})
                    print("Step: {:4d} time: {:.4f}, validation accuracy: {:.5f}, loss: {:.8f}".format(step, time.time()-start_time, acc, loss_value))
                    test_writer.add_summary(summary, step)
                # Save a checkpoint and evaluate the model periodically.
                if step % FLAGS.save_step == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

            coord.request_stop()
            coord.join(threads)

    print("done")

def run_testing():
    videos_placeholder, labels_placeholder, dropout_placeholder = placeholder_inputs()
    logits = []
    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for gpu_index in range(0, FLAGS.gpu_num):
            with tf.device('/gpu:%d' % gpu_index):
                print('/gpu:%d' % gpu_index)
                with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:
                    logit = c3d_model.inference_c3d(
                        videos_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                        dropout_placeholder,
                        FLAGS.batch_size
                    )
                    logits.append(logit)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
    logits = tf.concat(logits, 0)
    right_count = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), labels_placeholder), tf.int32))
    softmax_logits_op = tf.nn.softmax(logits)

    train_files = [os.path.join(FLAGS.training_data, f) for f in
                   os.listdir(FLAGS.training_data) if f.endswith('.tfrecords')]
    val_files = [os.path.join(FLAGS.validation_data, f) for f in
                 os.listdir(FLAGS.validation_data) if f.endswith('.tfrecords')]
    videos_op, labels_op = input_data.inputs(filenames=train_files,
                                       batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                                       num_epochs=1,
                                       num_threads=FLAGS.num_threads,
                                       num_examples_per_epoch=FLAGS.num_examples_per_epoch,
                                       shuffle=False
                                       )

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver(tf.trainable_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Session Restored from {}!'.format(ckpt.model_checkpoint_path))
    total_v = 0.0
    test_correct_num = 0.0
    try:
        while not coord.should_stop():
            videos, labels = sess.run([videos_op, labels_op])
            #write_video(videos, labels)
            feed = {videos_placeholder: videos, labels_placeholder: labels, dropout_placeholder: 1.0}
            right, softmax_logits= sess.run([right_count, softmax_logits_op], feed_dict=feed)
            test_correct_num += right
            total_v += labels.shape[0]
            print(softmax_logits.shape)
            print(tf.argmax(softmax_logits, 1).eval(session=sess))
            print(labels)
    except tf.errors.OutOfRangeError:
        print('Done testing on all the examples')
    finally:
        coord.request_stop()
    print('test acc:', test_correct_num / total_v, 'test_correct_num:', test_correct_num,
              'total_v:', total_v)
    coord.join(threads)
    sess.close()

def write_video(videos, labels):
    import cv2
    class_dict = {}
    with open('ucfTrainTestlist/classInd.txt', 'r') as f:
        for line in f:
            #print(line)
            words = line.strip('\n').split()
            class_dict[int(words[0]) - 1] = words[1]

    width, height = 112, 112
    for i in range(len(videos)):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output = '{}_{:d}.avi'.format(class_dict[labels[i]], i)
        #output = '{}_{:f}_{:d}.avi'.format(class_dict[labels[i]], sigmas[i], i)
        out = cv2.VideoWriter(output, fourcc, 1.0, (width, height), True)
        vid = videos[i]
        vid = vid.astype('uint8')
        for i in range(vid.shape[0]):
            frame = vid[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame.reshape(112, 112, 3)
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)
    run_training()
    #run_testing()

if __name__ == '__main__':
    tf.app.run()
