# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:59:10 2019

@author: zida
"""
import os 
import cv2
import numpy as np
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.models import Sequential
import keras
from keras.layers import Reshape

def normalization(image):
    return (image - image.min())/(image.max() - image.min())

def make_dataset(path,view):
    classes = ['bags', 'books', 'bottles', 'cups', 'pens', 'phones']
    print(path)
    view_1 = 0
    view_2 = 0
    view_3 = 0
    view_4 = 0
    view_5 = 0
    view_6 = 0
    record = np.zeros(100)
    i = 0
    for iterm in classes:
        print(path+iterm)
        files = os.listdir(path+iterm)
        for file_name in files:
            if 'view1' in file_name:
                image = cv2.imread(path+iterm+'/'+file_name)
                start = time.clock()
                image = cv2.resize(image,(224,224))
                if i < 100:
                    record[i] = time.clock()-start
                    i = i +1
                    print(i)
                image = normalization(image)
                view[0][view_1] = image
                view_1 = view_1 + 1
            elif 'view2' in file_name:
                image = cv2.imread(path+iterm+'/'+file_name)
                image = cv2.resize(image,(224,224))
                image = normalization(image)
                view[1][view_2] = image
                view_2 = view_2 + 1
            elif 'view3' in file_name:
                image = cv2.imread(path+iterm+'/'+file_name)
                image = cv2.resize(image,(224,224))
                image = normalization(image)
                view[2][view_3] = image
                view_3 = view_3 + 1
            elif 'view4' in file_name:
                image = cv2.imread(path+iterm+'/'+file_name)
                image = cv2.resize(image,(224,224))
                image = normalization(image)
                view[3][view_4] = image
                view_4 = view_4 + 1
            elif 'view5' in file_name:
                image = cv2.imread(path+iterm+'/'+file_name)
                image = cv2.resize(image,(224,224))
                image = normalization(image)
                view[4][view_5] = image
                view_5 = view_5 + 1
            elif 'view6' in file_name:
                image = cv2.imread(path+iterm+'/'+file_name)
                image = cv2.resize(image,(224,224))
                image = normalization(image)
                view[5][view_6] = image
                view_6 = view_6 + 1

def build_model( ):
    target_size = 224
    input_tensor = Input(shape=(target_size, target_size, 3))
    base_model = MobileNetV2(
        include_top=False,
         weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(target_size, target_size, 3),
        pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers
    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.25)(op)
    output_tensor = Dense(6, activation='softmax')(op)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model


def main(folds):
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    data = np.zeros((6,216,224,224,3))
    data[:,0:108,:,:,:] = make_dataset(folds[1],data[:,0:108,:,:,:])
    data[:,108:216,:,:,:] = make_dataset(folds[2],data[:,108:216,:,:,:])
    label = np.zeros(108)
    for i in range(0,108):
        label[i] = int(i/18)
    label = np.concatenate([label,label])
    data = np.swapaxes(data, 0, 1)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
    number = X_train.shape[0]*X_train.shape[1]
    filepath= folds[3]
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    Xtr = np.zeros((number,224,224,3))
    ytr = np.zeros((number))
    for i in range(0,6):
        Xtr[i*X_train.shape[0]:(i+1)*X_train.shape[0]] = X_train[:,i,:,:,:]
        ytr[i*X_train.shape[0]:(i+1)*X_train.shape[0]] = y_train

    ytr_onehot = keras.utils.to_categorical(ytr, 6)

    
    history = model.fit(Xtr, ytr_onehot,
                   batch_size=32,
                   epochs=50,
                   validation_split=0.5,
                   callbacks = callbacks_list,
                   shuffle=True)


if __name__== "__main__":
    main(sys.argv)