# coding:utf-8

import numpy as np
import argparse
import cv2
import tensorflow as tf
import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
from keras import backend as K

import train

# openCV -> keras 
def cvt_keras(img):
    # resize
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = x.reshape((1,) + x.shape)
    x /= 255
    return x

# message API用
def pred(img):
    classes = ['engineering_faculty', 'law_department']

    """ load model """
    pred_model = load_model('./trained_model.h5')
    
    print('load model')
    # kerasで読めるようにデータを加工
    img = cvt_keras(img)

    # 予測
    pred = pred_model.predict(img, batch_size=1)
    print(pred)
    score = np.max(pred)
    print(score)
    pred_label = np.argmax(pred)
    print(pred_label)

    # メモリ解放
    K.clear_session()
    tf.reset_default_graph()

    if pred_label == 0:
        return 0, score
    else:
        return 1 ,score

# テスト用
def main(args):
    img = load_img(args.imgpath, target_size=(224,224))
    img_array = img_to_array(img)/255
    img_array = img_array[None, ...]

    """ load model """
    pred_model = load_model('train_log/model_epoch100_imgsize224_batchsize16/trained_model.h5')
    pred_model.summary()

    # 予測
    pred = pred_model.predict(img_array, batch_size=1)
    print(pred)
    score = np.max(pred)
    print(score)
    pred_label = np.argmax(pred)
    print(pred_label)

    # メモリ解放
    K.clear_session()
    tf.reset_default_graph()
    return 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', '-i', type=str, default='')

    args = parser.parse_args()
    main(args)