from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import *
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential

def create_base_network():
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
    return model

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