from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import shutil


datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen_test = ImageDataGenerator(rescale=1.0/255.0)

train_it = datagen.flow_from_directory('8-multi-class_data/train/', class_mode='categorical', batch_size=32, target_size=(200,200))
test_it = datagen_test.flow_from_directory('8-multi-class_data/test/', class_mode='categorical', batch_size=32, target_size=(200,200))
valid_it = datagen_test.flow_from_directory('8-multi-class_data/val/', class_mode='categorical', batch_size=32, target_size=(200,200))

#모델 만들기
input_layer = Input(shape=(200,200,3))
x = Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', padding='same')(input_layer)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=128, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(filters=256, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=256, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=256, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=512, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(5,5))(x)
x = Flatten()(x)
x = Dense(4096,activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(2000,activation='relu')(x)
x = Dropout(0.25)(x)
output_layer = Dense(8, activation='softmax')(x)
model = Model(input_layer,output_layer)
model.summary()

## model = build_model(224,224)    이것도 하고 데이터젠할때 사이즈 맞춰서 바꾸기.
opt = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit_generator(train_it,steps_per_epoch=len(train_it), epochs=50,validation_data=valid_it,validation_steps=len(valid_it))

print('Evaluate')
scores = model.evaluate_generator(test_it, steps=len(test_it))
print("%s:%.2f%%" %(model.metrics_names[1], scores[1]*100))


#모델 저장
save_dir = os.path.join(os.getcwd(), 'saved model')
model_name = 'multiclass_model_test115.h5'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)