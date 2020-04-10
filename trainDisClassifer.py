#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.applications.mobilenetv2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import random
import os
from keras.optimizers import SGD,Adam
import cv2
import numpy as np
import os
import sys
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.model_selection import train_test_split
from skimage.util import random_noise
from keras.layers import Reshape


def preprocess(image):
    return (image-image.min())/(image.max()-image.min())

def rgb2gray(rgb):
  image = np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
  image = preprocess(image)
  return image

def get_Fourier(image):
    f = np.fft.fft2(image)
    fshif = np.fft.fftshift(f)
    s1 = np.log(20+np.abs(fshif))
    s1 = preprocess(s1)
    return s1

def motion_blur(image, degree):
    if degree != 0:
      angle = random.randint(1,360)
      M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
      motion_blur_kernel = np.diag(np.ones(degree))
      motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

      motion_blur_kernel = motion_blur_kernel / degree
      blurred = cv2.filter2D(image, -1, motion_blur_kernel)
      cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
      blurred = np.array(blurred, dtype=np.uint8)
      if blurred.mean()>1:
        blurred = blurred/255
    else: 
      blurred = image
    return blurred

def gaussion_noise(image,k):
    image = image/255
    k = k /1000
    image = random_noise(image, mode="gaussian", var = k)
    if image.mean()>1:
        image/255
    return image

def gaussion_blur(image, k):
    if k!=0:
      k = 2*k+1
      image = cv2.GaussianBlur(image,(k,k),0)

    if image.mean()>1:
      image = image/255
      return image


def main(path):
	folder = os.path.exists('./weights')
	if not folder:
		os.makedirs('./weights') 
	images = os.listdir(path)

	X_train = np.zeros((len(images),224,224,3))
	X_train_F = np.zeros((len(images),224,224,1))
	type_label_train = np.zeros(len(images))
	i = 0
	for files in images:
		images = cv2.imread(path+files)
		images = cv2.resize(images,(224,224))
		a = random.randint(0,3)
		if a == 1:
			X_train[i] = motion_blur(images, random.randint(5,40))
			X_train_F[i,:,:,0] = get_Fourier(rgb2gray(X_train[i]))
		elif a == 2:
			X_train[i] = gaussion_blur(images , random.randint(3,20))
			X_train_F[i,:,:,0] = get_Fourier(rgb2gray(X_train[i]))
		elif a == 3:
			X_train[i] = gaussion_noise(images, random.randint(5,40))
			X_train_F[i,:,:,0] = get_Fourier(rgb2gray(X_train[i]))
		else:
			X_train_F[i,:,:,0] = get_Fourier(rgb2gray(images))
		type_label_train[i] = a
		i = i + 1

	y_train = keras.utils.to_categorical(type_label_train, 4)
	batch_size = 32
	num_classes = 4
	epochs = 30
	filepath="./weight/type_model.hdf5"
	checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
	callbacks_list = [checkpoint]

	model = Sequential()
	model.add(Conv2D(32, (3, 3),strides=(2, 2), padding='valid',
	                     input_shape=(224,224,1)))
	model.add(Activation('relu'))
	model.add(Conv2D(16, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	    
	model.add(Conv2D(1, (1, 1), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Reshape((1, 729)))
	model.add(Flatten())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4))
	model.add(Activation('softmax'))

	# Let's train the model
	model.compile(loss='categorical_crossentropy',
	              optimizer=Adam(lr = 0.0001),
	              metrics=['accuracy'])
	model.summary()

	model.fit(X_train_F, y_train,
	              batch_size=batch_size,
	              epochs=100,
	              validation_split=0.5,
	              callbacks = callbacks_list,
	              shuffle=True)



if __name__== "__main__":
    main(sys.argv[1])
