import tensorflow as tf
import datetime
import os

flags = tf.app.flags


_gamma = 2.0
_lambda = 0.5
_mode = 'SuppressingMostConfident'
N = 8

# Here resampling means restarting
resampling = True
l1_loss = True
avg_replicate = True
monitor_budget = True
monitor_utility = True
lambda_decay = True
residual = True

isResampling = lambda bool: "Resample" if bool else "NoResample"
isL1Loss = lambda bool: "L1Loss" if bool else "NoL1Loss"
isAvgReplicate = lambda bool: "AvgReplicate" if bool else "NoAvgReplicate"
isMonitorBudget = lambda bool: "MonitorBudget" if bool else "NoMonitorBudget"
isMonitorUtility = lambda bool: "MonitorUtility" if bool else "NoMonitorUtility"
isLambdaDecay = lambda bool: "LambdaDecay" if bool else "NoLambdaDecay"
isResidual = lambda bool: "UseResidual" if bool else "NoUseResidual"


log_dir = 'tensorboard_events/' + isL1Loss(l1_loss) + isLambdaDecay(lambda_decay) + isAvgReplicate(avg_replicate) + isMonitorBudget(monitor_budget) + isMonitorUtility(monitor_utility) + isResampling(resampling) + '{}_'.format(N) + '{}_'.format(_gamma) + '{}_'.format(_lambda) +  _mode + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
summary_dir = 'summaries/'  + isL1Loss(l1_loss) + isLambdaDecay(lambda_decay) + isAvgReplicate(avg_replicate) + isMonitorBudget(monitor_budget) + isMonitorUtility(monitor_utility) + isResampling(resampling) + '{}_'.format(N) + '{}_'.format(_gamma) + '{}_'.format(_lambda) + _mode + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
eval_vis_dir = 'eval_vis/' + isL1Loss(l1_loss) + isLambdaDecay(lambda_decay) + isAvgReplicate(avg_replicate) + isMonitorBudget(monitor_budget) + isMonitorUtility(monitor_utility) + isResampling(resampling) + '{}_'.format(N) + '{}_'.format(_gamma) + '{}_'.format(_lambda) + _mode + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

# Basic model parameters as external flags.
flags.DEFINE_string('log_dir', log_dir, 'Directory where to write the tensorboard events')
flags.DEFINE_string('summary_dir', summary_dir, 'Directory where to write the summary')
flags.DEFINE_string('eval_vis_dir', eval_vis_dir, 'Directory where to write the visualization of the evaluation frame')

flags.DEFINE_string('checkpoint_dir', 'checkpoint/checkpoint_{}/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(isResidual(residual),
                                    isL1Loss(l1_loss), isLambdaDecay(lambda_decay), isAvgReplicate(avg_replicate),
                                    isMonitorBudget(monitor_budget), isMonitorUtility(monitor_utility), isResampling(resampling),
                                    N,_gamma, _lambda), 'Directory where to read/write model checkpoints')
flags.DEFINE_string('saved_checkpoint_dir', 'checkpoint/checkpoint_{}/saved_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(isResidual(residual),
                                    isL1Loss(l1_loss), isLambdaDecay(lambda_decay), isAvgReplicate(avg_replicate),
                                    isMonitorBudget(monitor_budget), isMonitorUtility(monitor_utility), isResampling(resampling),
                                    N,_gamma, _lambda), 'Directory where to read/write model checkpoints')

flags.DEFINE_string('visualization_dir', 'visualization/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(isResidual(residual),
                                    isL1Loss(l1_loss), isLambdaDecay(lambda_decay), isAvgReplicate(avg_replicate),
                                    isMonitorBudget(monitor_budget), isMonitorUtility(monitor_utility), isResampling(resampling),
                                    N,_gamma, _lambda), 'Directory where to write degradation visualization')

flags.DEFINE_string('budget_evaluation', 'checkpoint/budget_evaluation_wholefinetune_'  + isResampling(resampling) + '{}_'.format(N) + '{}'.format(_gamma) + "/", 'Directory where to read/write model checkpoints')
flags.DEFINE_string('utility_models', 'checkpoint/utility', 'Directory where to read/write model checkpoints')
flags.DEFINE_string('degradation_models', 'checkpoint/degradation/{}'.format(isResidual(residual)), 'Directory where to read/write model checkpoints')
flags.DEFINE_string('budget_models', 'checkpoint/budget', 'Directory where to read/write model checkpoints')

flags.DEFINE_string('whole_pretraining', '../checkpoint/whole_pretraining/{}'.format(isResidual(residual)), 'Directory where to read/write model checkpoints')

flags.DEFINE_float('_gamma', _gamma, 'Hyperparameter for the weighted combination of utility loss and budget loss')
flags.DEFINE_float('_lambda', _lambda, 'Hyperparameter for the weight of L1 loss')

flags.DEFINE_float('degradation_lr', 1e-4, 'Learning rate for the degradation model')
flags.DEFINE_float('utility_lr', 1e-5, 'Learning rate for the utility model')
flags.DEFINE_float('budget_lr', 1e-2, 'Learning rate for the budget model')

flags.DEFINE_float('largest_gap', 0.15, 'Largest gap for saving the model')
flags.DEFINE_float('highest_util_acc_val', 0.85, 'Monitoring the validation accuracy of the utility task')
flags.DEFINE_float('highest_budget_acc_train', 0.99, 'Monitoring the training accuracy of the budget task')
flags.DEFINE_float('lowest_budget_acc_val', 0.5, 'Monitoring the validation accuracy of the budget task')

flags.DEFINE_integer('resample_step', 200, 'Number of steps for resampling model')
flags.DEFINE_integer('retraining_step', 1000, 'Number of steps for retraining sampled model')

flags.DEFINE_integer('NBudget', N, 'Number of budget models')
flags.DEFINE_boolean('use_resampling', resampling, 'Whether to use resampling model')

flags.DEFINE_boolean('use_crop', True, 'Whether to use crop when reading video in the input pipeline')
flags.DEFINE_boolean('use_random_crop', False, 'Whether to use random crop when reading video in the input pipeline')
flags.DEFINE_boolean('use_center_crop', True, 'Whether to use center crop when reading video in the input pipeline')
flags.DEFINE_boolean('use_avg_replicate', avg_replicate, 'Whether to replicate the 1 channel by averaging the 3 channels degradation module output')
flags.DEFINE_boolean('use_l1_loss', l1_loss, 'Whether to use the l1 loss regularizer')
flags.DEFINE_boolean('use_monitor_budget', monitor_budget, 'Whether to monitor the budget task')
flags.DEFINE_boolean('use_monitor_utility', monitor_utility, 'Whether to monitor the utility task')
flags.DEFINE_boolean('use_lambda_decay', lambda_decay, 'Whether to use lambda decay')
flags.DEFINE_boolean('use_residual', residual, 'Whether to use the residual net')

flags.DEFINE_integer('n_minibatches', 20, 'Number of mini-batches')
flags.DEFINE_integer('n_minibatches_eval', 40, 'Number of mini-batches')
flags.DEFINE_string('mode', _mode, 'Training mode when updating the filtering model')

FLAGS = flags.FLAGS