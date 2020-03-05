import numpy as np
import cv2
import os
import sys
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.models import Sequential
import keras
from keras.layers import Reshape
from skimage.util import random_noise
import random
from keras_preprocessing.image import ImageDataGenerator


def pristine(image):
    return image/255

def motion_blur(image):
    blurred = image
    if random.randint(1,1)== 1:
        degree = random.randint(0,40)
        if degree != 0:
            angle = random.randint(1,360)
            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.ones(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
            motion_blur_kernel = motion_blur_kernel / degree
            blurred = cv2.filter2D(image, -1, motion_blur_kernel)
            cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    if blurred.mean()>1:
        blurred = blurred/255
    return blurred

def gaussion_noise(image):
    if random.randint(1,1)== 1:
        k = random.randint(0,40)
        k = k /1000
        if image.mean()>1:
            image = image/255
        image = random_noise(image, mode="gaussian", var = k)
    if image.mean()>1:
        image = image/255
    return image

def gaussion_blur(image):
    if random.randint(1,1)==1:
        k = random.randint(0,20)
        if k!=0:
            k = 2*k+1
            image = cv2.GaussianBlur(image,(k,k),0)
    if image.mean()>1:
        image = image/255
    return image

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
    output_tensor = Dense(257, activation='softmax')(op)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model

def main(argv):
    folder = os.path.exists('./weights')
    if not folder:
        os.makedirs('./weights') 
    batch_size = 32
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(lr = 0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if argv[1] == 'pristin':
        train_datagen = ImageDataGenerator(preprocessing_function=pristine)
        validation_datagen = ImageDataGenerator(preprocessing_function=pristine)
        test_datagen = ImageDataGenerator(preprocessing_function=pristine)
        filepath="./weights/pristine_expert.hdf5"
    elif argv[1] == 'MB':
        train_datagen = ImageDataGenerator(preprocessing_function=motion_blur)
        validation_datagen = ImageDataGenerator(preprocessing_function=motion_blur)
        test_datagen = ImageDataGenerator(preprocessing_function=motion_blur)
        filepath="./weights/motion_blur_expert.hdf5"
    elif argv[1] == 'GB':
        train_datagen = ImageDataGenerator(preprocessing_function=gaussion_blur)
        validation_datagen = ImageDataGenerator(preprocessing_function=gaussion_blur)
        test_datagen = ImageDataGenerator(preprocessing_function=gaussion_blur)
        filepath="./weights/Gaussian_blur_expert.hdf5"
    else:
        train_datagen = ImageDataGenerator(preprocessing_function=gaussion_noise)
        validation_datagen = ImageDataGenerator(preprocessing_function=gaussion_noise)
        test_datagen = ImageDataGenerator(preprocessing_function=gaussion_noise)
        filepath="./weights/Gaussian_noise_expert.hdf5"

    train_generator = train_datagen.flow_from_directory('train', target_size=(224,224),batch_size=batch_size, class_mode = 'categorical')
    validation_generator = validation_datagen.flow_from_directory('validation', target_size=(224,224),batch_size=batch_size, class_mode = 'categorical')
    test_generator = test_datagen.flow_from_directory('test', target_size=(224,224),batch_size=batch_size, class_mode = 'categorical')

    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    
    if argv[1] != 'pristin':
        model.load_weights('./weights/pristin_expert.hdf5')

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
    history = model.fit_generator(train_generator,
                        steps_per_epoch= STEP_SIZE_TRAIN ,
                        epochs=100,  
                        validation_data=validation_generator,
                        validation_steps=STEP_SIZE_VALID,
                        callbacks = callbacks_list)

if __name__== "__main__":
    main(sys.argv)
