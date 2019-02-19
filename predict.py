# coding:utf-8

import numpy as np
import cv2
import keras

# openCV -> keras 
def cvt_keras(img):
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = x.transpose((2,0,1))
    x = x.reshape((1,) + x.shape)
    x /= 255
    return x

def pred(img):
    # kerasで読めるようにデータを加工
    img = cvt_keras(img)

    return 