# coding:utf-8

import numpy as np
import argparse
import cv2
import keras
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model

import train


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