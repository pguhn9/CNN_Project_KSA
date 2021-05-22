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


#loss그래프 그리기
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

#acuuracy 그래프 그리기
def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'val'], loc=0)

save_dir = os.path.join(os.getcwd(), "saved_model")
print(os.getcwd())
print(save_dir)
model_name = "Keras_cifar10_aug_trained_model.h5"



if not os.path.isdir(save_dir):
    os.mkdirs(save_dir)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#입력 이미지 사이즈 정보
img_rows, img_cols, channel=32,32,3

#filter의 갯수
nb_filters = 64

if K.image_dim_ordering()=='th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0],3,img_rows,img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0],img_rows, img_cols,3)
    X_test = X_test.reshape(X_test.shape[0],img_rows, img_cols,3)
    input_shape = (img_rows, img_cols,3)


CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck'])


NUM_CLASSESE = 10


#type 재설정
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#255로 나누어 정규화
X_train = X_train/255
X_test = X_test/255


y_train = to_categorical(y_train, NUM_CLASSESE)
y_test = to_categorical(y_test, NUM_CLASSESE)


batch_size = 128
num_classes = 10
epochs =7



input_tensor=Input(shape=input_shape)
x = BatchNormalization()(input_tensor)
x = Conv2D(nb_filters, kernel_size=(3,3), strides=1, padding='same', name='Conv1')(x)
x = Activation('relu', name='relu_1')(x)
x = Dropout(0.2)(x)
#x = LeakyReLU()(x)
x = Conv2D(nb_filters, kernel_size=(3,3), strides=2, padding='same', name='Conv2')(x)
x = Activation('relu', name='relu_2')(x)
x = Dropout(0.2)(x)
x = Conv2D(nb_filters, kernel_size=(3,3), strides=1, padding='same', name='Conv3')(x)
x = Activation('relu', name='relu_3')(x)
x = Dropout(0.2)(x)
x = Conv2D(nb_filters, kernel_size=(3,3), strides=2, padding='same', name='Conv4')(x)
x = Activation('relu', name='relu_4')(x)
x = Dropout(0.2)(x)
x = MaxPooling2D(pool_size=(2,2), name='pool_1')(x)
x = Flatten()(x)
#x = Activation('relu', name='relu_4')(x)
x = Dense(units=128, name='hidden_1')(x)
x = BatchNormalization()(x)
x = Activation('relu', name='relu_5')(x)
x = Dense(units=128, name='hidden_2')(x)
x = LeakyReLU()(x)
x = Dense(units=num_classes, name='hidden_3')(x)
output_tensor = Activation('softmax', name='output_tensor')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)



model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()



#data aug
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

# 모델 학습(aug적용)
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=64),
                              steps_per_epoch=X_train.shape[0]/64, epochs=7, verbose=1, #5만개인데 배치사이즈만큼 돌리니까 이를 나눠서 이만큼.
                              validation_data=(X_test, y_test), workers=4) # validation_data를 통해 하이퍼파라미터 튜닝을 위함. 80 5 15


#model 학습(aug미적용)
#history = model.fit(X_train, y_train, batch_size=batch_size,
#                    epochs=epochs, verbose=1, validation_split=0.2)#하이퍼파라미터 튜닝


#모델 저장
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

#predict
print("Test start")
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('\nTest loss:',score[0])
print('test Accuracy:',score[1])


#학습된 loss값과, accuracy 보기 위한 그래프
plot_loss(history)
plt.show()
plot_acc(history)
plt.show()


#학습모델 결과를 이미지로 출력하기
CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck']) #전체적인 라벨

preds = model.predict(X_test)
preds_single = CLASSES[np.argmax(preds, axis=-1)]
actual_single = CLASSES[np.argmax(y_test, axis=-1)]

#plot 그림 그리는 부분
n_to_show = 10
indices = np.random.choice(range(len(X_test)), n_to_show)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = X_test[idx]
    ax = fig.add_subplot(1, n_to_show, i+1)
    ax.axis("off")
    ax.text(0.5, -0.35, "pred = "+str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, "act = " + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)

plt.show()