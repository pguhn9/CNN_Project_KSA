from sklearn import datasets
from tensorflow.keras.models import Sequential #함수 형태로 해도됨. 이번엔 시퀀셜  Model
from tensorflow.keras.layers import Dense
import  matplotlib.pyplot as plt
from keras.utils import to_categorical #인덱스를 어떻게 해줄껀지. 카테고리컬
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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



iris = datasets.load_iris()
print(iris)



X = iris.data
Y = iris.target
### dataset_y = to_categorical(iris.target)
### dataset_x = iris.data
### dataset_x, dataset_y = shuffle(dataset_x, dataset_y)


print(X)
print(Y)
X_sfd, Y_sfd = shuffle(X, Y)
print(X_sfd)
print(Y_sfd)

X_train = X_sfd[:120]
Y_train = Y_sfd[:120]
X_test = X_sfd[120:150] ###[120:] 하면 끝까지 됨.
Y_test = Y_sfd[120:150]

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)




#하이퍼파라미터 설정
batch_size = 60
num_classes = 3
epochs =12

#원핫
y_train = to_categorical(Y_train, num_classes)
y_test = to_categorical(Y_test, num_classes)

print(y_train)
print(y_test)

# 모델 쌓기
model = Sequential()
model.add(Dense(128, activation='relu', name='input_tensor', input_shape=(4,)))
model.add(Dense(128, activation='relu', name='hidden_1'))
model.add(Dense(128, activation='relu', name='hidden_2'))
model.add(Dense(128, activation='relu', name='hidden_3'))
model.add(Dense(num_classes, activation='softmax', name='output_tensor'))

#모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) ### 소프트맥스이기때문에 케테고리컬 크로스엔트로피


model.summary() ### 굳이 안해도됨. 어떻게 구성되었는지 보는 것.


#model 학습
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2)#하이퍼파라미터 튜닝  ### valdattion_split 안해도 됨.원래는 있어야하는데. 지금은 직접 넣음 테스트 데이터  (X_test, y_test)

#predict
print("Test start")
score = model.evaluate(X_test, y_test, batch_size=batch_size) ### barbose= 진행률을 보는 파라미터. 0하면 안보임.
print('\nTest loss:', score[0])
print('test Accuracy:', score[1])


#학습된 loss값과, accuracy 보기 위한 그래프
plot_loss(history)
plt.show()
plot_acc(history)
plt.show()
