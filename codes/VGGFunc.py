from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D, BatchNormalization, LeakyReLU
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.regularizers import l2


def build_model_16(img_hight, img_width, img_channel, class_count, weight_decay):
    input_tensor = Input(shape=(img_hight, img_width, img_channel))
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv1')(input_tensor)
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool_1')(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv3')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv4')(x)
    x = MaxPooling2D(pool_size=(2, 2),  name='pool_2')(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv5')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv6')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv7')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv8')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool_3')(x)

    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv9')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv10')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv11')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv12')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool_4')(x)

    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv13')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv14')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv15')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay), activation='relu', name='Conv16')(x)
    x = MaxPooling2D(pool_size=(2, 2),  name='pool_5')(x)

    x = Flatten()(x)
    x = Dense(units=4096, name='hidden_1')(x)
    x = Activation('relu')(x)
    x = Dense(units=4096, name='hidden_2')(x)
    x = Activation('relu')(x)
    x = Dense(units=class_count, name='hidden_3')(x)
    output_tensor = Activation('softmax', name='output_tensor')(x)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.summary()

    return model


#model_build = build_model_16(224, 224, 3, 1000, 1e-4)
#model_build.summary()