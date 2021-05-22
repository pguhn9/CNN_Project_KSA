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

def build_model(img_hight, img_width, img_channel, class_count, weight_decay):
    input_tensor = Input(shape=(img_hight, img_width, img_channel))
    x = Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name='Conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, name='pool_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name='Conv2')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, name='pool_2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name='Conv3')(x)
    x = Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name='Conv4')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), activation='relu', name='Conv5')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, name='pool_3')(x)
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

model_build = build_model(227, 227, 3, 1000, 1e-4)
model_build.summary()








# x = Activation('relu', name='relu_1')(x)
# x = BatchNormalization()(x)
# x = Conv2D(nb_filters, kernel_size=(3,3), strides=1, padding='same', name='Conv2')(x)
# x = Activation('relu', name='relu_2')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D(pool_size=(2,2), name='pool_1')(x)
# #1층
# x = Conv2D(nb_filters*2, kernel_size=(3,3), strides=1, padding='same', name='Conv3')(x)
# x = Activation('relu', name='relu_3')(x)
# x = BatchNormalization()(x)
# x = Conv2D(nb_filters*2, kernel_size=(3,3), strides=1, padding='same', name='Conv4')(x)
# x = Activation('relu', name='relu_4')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D(pool_size=(2,2), name='pool_2')(x)
# #2층
# x = Conv2D(nb_filters*2*2, kernel_size=(3,3), strides=1, padding='same', name='Conv5')(x)
# x = Activation('relu', name='relu_5')(x)
# x = BatchNormalization()(x)
# x = Conv2D(nb_filters*2*2, kernel_size=(3,3), strides=1, padding='same', name='Conv6')(x)
# x = Activation('relu', name='relu_6')(x)
# x = BatchNormalization()(x)
# x = Conv2D(nb_filters*2*2, kernel_size=(3,3), strides=1, padding='same', name='Conv7')(x)
# x = Activation('relu', name='relu_7')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D(pool_size=(2,2), name='pool_3')(x)
# #x = LeakyReLU()(x)
# x = BatchNormalization()(x)
# x = Conv2D(nb_filters*2*2, kernel_size=(3,3), strides=2, padding='same', name='Conv8')(x)
# x = Activation('relu', name='relu_8')(x)
# x = Dropout(0.2)(x)
# x = Conv2D(nb_filters*2*2, kernel_size=(3,3), strides=1, padding='same', name='Conv9')(x)
# x = Activation('relu', name='relu_9')(x)
# x = Dropout(0.2)(x)
# x = Conv2D(nb_filters*2*2, kernel_size=(3,3), strides=2, padding='same', name='Conv10')(x)
# x = Activation('relu', name='relu_10')(x)
# x = Dropout(0.2)(x)
# #x = MaxPooling2D(pool_size=(2,2), name='pool_4')(x)
# x = Flatten()(x)
# #x = Activation('relu', name='relu_4')(x)
# x = Dense(units=128, name='hidden_1')(x)
# x = BatchNormalization()(x)
# x = Activation('relu', name='relu_11')(x)
# x = Dense(units=128, name='hidden_2')(x)
# x = LeakyReLU()(x)
# x = Dense(units=num_classes, name='hidden_3')(x)
# output_tensor = Activation('softmax', name='output_tensor')(x)
# model = Model(inputs=input_tensor, outputs=output_tensor)