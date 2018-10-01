import numpy as np
import cv2
import os
from tqdm import tqdm
import copy


def get_SBU_action(path):
    return path.split('/')[-2]


def get_SBU_actor(path):
    return path.split('/')[-3]


def get_SBU_setting(path):
    return path.split('/')[-1]


def write_video(X, Y_action, Y_actor, Y_setting):
    width, height = 640, 480
    for i in range(len(X)):
        print(os.getcwd())
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # Be sure to use lower case
        output = "/home/wuzhenyu_sjtu/DAN_sbu/SBU/SBU_videos/clean_version/{}_{}_{}.avi".format(Y_actor[i], Y_action[i], Y_setting[i])
        out = cv2.VideoWriter(output, fourcc, 10.0, (width, height), True)
        vid = X[i]
        vid = vid.astype('uint8')
        print(vid.shape)
        print(output)
        for i in range(vid.shape[0]):
            frame = vid[i]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame.reshape(480, 640, 3)
            #print(frame)
            out.write(frame)
        out.release()
        cv2.destroyAllWindows()

from skimage import io
from skimage.transform import resize
import os
import re

path = "/home/wuzhenyu_sjtu/DAN_sbu/SBU/SBU_Kinect/clean_version"
print(path)
path = os.path.normpath(path)
print(path)
X = []
Y_action = []
Y_actor = []
Y_setting = []
for root, dirs, files in os.walk(path):
    depth = root[len(path):].count(os.path.sep)
    if depth == 3:
        print(root)
        files = [file for file in files if file.startswith("mono-depth")]
        files.sort()
        actor = get_SBU_actor(root)
        action = get_SBU_action(root)
        setting = get_SBU_setting(root)
        framelst = []
        for file in files:
            filename = os.path.join(root, file)
            frame = io.imread(filename)
            framelst.append(frame)
        X.append(np.asarray(framelst))
        Y_action.append(action)
        Y_actor.append(actor)
        Y_setting.append(setting)

write_video(X, Y_action, Y_actor, Y_setting)