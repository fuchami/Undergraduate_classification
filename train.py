# coding:utf-8

import argparse
import h5py

import numpy as np
import keras
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.applications.mobilenetv2 import MobileNetV2
from keras.models import Model
from keras.layers import Input, Dense, Dropout

import load, tools

def mobileNet_model(input_shape, classes):

    Input_shape = Input(shape=input_shape)
    mobilenet = MobileNetV2(include_top=False, 
        alpha=1.0, weights='imagenet', pooling='avg')
    mobilenet.trainable = False
    mobilenet.summary()

    x_in = Input_shape
    x = mobilenet(x_in)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64 , activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(classes, activaton='softmax')(x)

    model = Model(x_in, x)
    model.summary()

    return model

def main(args, classes):

    """ log params """
    para_str = 'model_epoch{}_imgsize{}_batchsize{}/'.format(
        args.epochs, args.imgsize, args.batchsize)
    para_path = './train_log/' + para_str
    if not os.path.exitst(para_path +'/'):
        os.makedirs(para_path + '/')

    """ define callback """
    base_lr = 1e-3
    lr_decay_rate = 1/3
    lr_steps = 4
    reduce_lr = LearningRateScheduler(lambda ep: float(base_lr*lr_decay_rate**(ep * lr_steps// args.epochs)), verbose=1)

    callbacks = []
    callbacks.append(reduce_lr)

    """ load image using image data generator """
    train_generator, valid_generator = load.generator(args, classes)

    """ build model """
    input_shape = (args.imgsize, args.imgsize, 3)
    mobile_model = mobileNet_model(input_shape, len(classes))

    opt = SGD(lr=base_lr, momentum=0.9, decay=1e-6, nesterov=True)
    plot_model(mobile_model, to_file='./model.png', show_shapes=True)
    mobile_model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])
    
    history = mobile_model.fit_generator(
        generator = train_generator,
        steps_per_epoch = 500// args.batchsize,
        nb_epoch = args.epochs,
        callbacks = callbacks,
        validation_data = valid_generator,
        validation_steps=1)
    
    tools.plot_history(history, para_str, para_path)
    mobile_model.save(para_path '/trained_model.h5')


    return

if __name__ == "__main__":
    classes = ['engineering_faculty', 'law_department']

    parser = argparse.ArgumentParser(description='train mobileNet')
    parser.add_argument('--trainpath', type=str, default='')
    parser.add_argument('--validpath', type=str, default='')
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--imgsize', '-i', type=int, default=244)
    parser.add_argument('--batchsize', '-b', type=int, default=32)

    args = parser.parse_args()
    main(args, classes)