from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np

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

#하이퍼파라미터 설정
batch_size = 128
num_classes = 10
epochs =12

#데이터 로드
(X_train, Y_train),(X_test, Y_test) = mnist.load_data()

#mnist 데이터 이미지 추출 부분
# plt.figure(filesize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(np.reshape(X_train[i]),[28,28],cmap='Grays')
#     plt.xlabel(Y_train[i])
# plt.show()


L,W,H = X_train.shape

#2차원을 1차원에 넣기위한 reshape
X_train = X_train.reshape(60000,W*H)
X_test = X_test.reshape(10000,W*H)

#type 재설정
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#255로 나누어 정규화
X_train = X_train/255
X_test = X_test/255

#라벨(원핫인코딩)
# print("Y_train_ori:{}".format(Y_train[:5])) #test(Y_train_ori)
y_train = to_categorical(Y_train, num_classes)
# print("Y_train_after:{}".format(Y_train[:5])) #test(Y_train_after)
y_test = to_categorical(Y_test, num_classes)


#네트워크 정의
#1. function()
input_tensor = Input(shape=(784,),name='input_tensor')
hidden_1 = Dense(units=256, activation='relu', name='hidden_1')(input_tensor)
hidden_2 = Dense(units=256, activation='relu', name='hidden_2')(hidden_1)
hidden_3 = Dense(units=256, activation='relu', name='hidden_3')(hidden_2)
hidden_4 = Dense(units=256, activation='relu', name='hidden_4')(hidden_3)
output_tensor = Dense(num_classes, activation='softmax', name='output_tensor')(hidden_4)
model = Model(inputs=input_tensor, outputs=output_tensor)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary() #model이 어떻게 구성되어있는지 확인가능

#2. sequential()
# model = Sequential()
# model.add(Dense(256, activation='relu', name='input_tensor', input_shape=(784,)))
# model.add(Dense(activation='relu', name='hidden_1'))
# model.add(Dense(activation='relu', name='hidden_2'))
# model.add(Dense(activation='relu', name='hidden_3'))
# model.add(Dense(num_classes, activation='softmax', name='output_tensor'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

#model 학습
history = model.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_split=0.2)#하이퍼파라미터 튜닝

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