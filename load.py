# coding:utf-8
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def generator(args, classes):

    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2) 

    valid_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        args.trainpath,
        target_size=(args.imgsize, args.imgsize),
        color_mode='rgb',
        classes=classes,
        class_mode = 'categorical',
        batch_size=args.batchsize,
        shuffle=True)
        
    valid_generator = valid_datagen.flow_from_directory(
        args.validpath,
        target_size=(args.imgsize, args.imgsize),
        color_mode='rgb',
        classes=classes,
        class_mode = 'categorical',
        shuffle=False)
    
    return train_generator, valid_generator