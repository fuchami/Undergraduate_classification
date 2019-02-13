# coding:utf-8
import cv2
import numpy
import sys, os, glob
import matplotlib.pyplot as plt
import argparse

def main(args):
    # 学部をリスト化
    class_path_list = glob.glob(args.srcpath + '*')
    print('class_path_list: ', class_path_list)

    # Haar-like分類の読み込み
    cascade = cv2.CascadeClassifier(args.cascadepath)

    for class_path in class_path_list:
        """ 保存先の確保 """
        tar_img_path = args.tarpath + os.path.basename(class_path) +'/'
        if not os.path.exists(tar_img_path):
            os.makedirs(tar_img_path)

        cnt = 0
        
        img_path_list = glob.glob(class_path + '/*.JPG')
        for img_path in img_path_list:
            print('img_path: ', img_path)
            src_img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)

            facerect = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=1, minSize=(180,180))
            if len(facerect)>0:
                for rect in facerect:
                    face_img = src_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

                    cv2.imwrite('%simg%04d.jpg' % (tar_img_path, cnt), face_img)
                    print('saved %simg%04d.jpg' % (tar_img_path, cnt))
                    cnt +=1
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcpath', '-s', type=str, default='/Users/yuuki/Desktop/GRADUATION_ALBUM/ORIGINAL/')
    parser.add_argument('--tarpath', '-t', type=str, default='/Users/yuuki/Desktop/GRADUATION_ALBUM/cropped/')
    parser.add_argument('--cascadepath', '-c', type=str, default='./haarcascade_frontalface_alt.xml')
    

    args = parser.parse_args()
    main(args)
