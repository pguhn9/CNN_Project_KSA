from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation, Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import numpy as np
from keras import backend as K



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


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

CLASSES = np.array(['airplane', 'automobile', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck'])
actual_single = CLASSES[y_train]
plt.imshow(x_test[20], interpolation="bicubic")
tmp = "Label:" + str(actual_single[20])
plt.title(tmp, fontsize=30)
plt.tight_layout()
plt.show()

print(x_train.shape)
L,W,H,C = x_train.shape

NUM_CLASSESE = 10

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, NUM_CLASSESE)
y_test = to_categorical(y_test, NUM_CLASSESE)

##L,W,H = x_train.shape
#하이퍼파라미터 설정
batch_size = 128
num_classes = 10
epochs =4




# 모델 정의 부분 (DNN) -> unt = 200, dense 총 3개로 이용하기
#input_layer = Input((shape=3072,), name='input_tensor')
input_layer = Input(shape=(32, 32, 3))
x = Flatten()(input_layer)
x = Dense(units=200, activation='relu')(x)
x = Dense(units=200, activation='relu')(x)
output_layer = Dense(units=10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)

#######################

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#########################

# 모델 훈련
history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_split=0.2)

#########################3

# 모델 평가
print("Test start")
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print('\nTest loss:', score[0])
print('test Accuracy:', score[1])

plot_loss(history)
plt.show()
plot_acc(history)
plt.show()


###########################3
print("Test2 start")
score2 = model.predict(x_test, y_test)
print('\nTest loss2:', score2[0])
print('test Accuracy2:', score2[1])