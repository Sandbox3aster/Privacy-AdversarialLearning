
# coding: utf-8

# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
from SBU.VideoReader import SBUReader
import resource

def video_processing(videos_dir):
    vreader = SBUReader(depth=16, sigma=1.0, ksize=4)
    #X, Y_action, Y_actor = vreader.loaddata_LR(videos_dir, train=False)

    train_lst, val_lst, test_lst = vreader.loaddata_HR(videos_dir)
    X, Y_action, Y_actor = train_lst[0], train_lst[1], train_lst[2]

    print('X shape:{}\nY_action shape:{}\nY_actor shape:{}'.format(X.shape,
                                                                    Y_action.shape, Y_actor.shape))

    return train_lst, val_lst, test_lst


def convert_to_npy(folder):
    path = os.path.join('SBU_videos/', folder)
    print(path)
    videos, _, _ = video_processing(path)
    np.save('eval.npy', videos[0])

convert_to_npy('evaluation')
eval = np.load('eval.npy')
print(eval.shape)

