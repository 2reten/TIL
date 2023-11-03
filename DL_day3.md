#  CNN과 DNN


```python
"""
Computer Vision Problem 
엣지 검출(ex: 사람이 서있나 ) -> 객체 일부 -> 완전객체 감지

*엣지 검출 (vertical edge detection) : 6 * 6 행렬일 경우 3*3(filter)을 이용해 6*6행렬의 각 요소(3*3)끼리 곱을 더해서 
                                       첫번째 칸에 넣어주고 한칸씩 이동하면서 3*3(filter)을 요소(3*3끼리 곱한값의 합으로 4*4행렬 채워진다.
https://076923.github.io/posts/Python-opencv-29/ :open-cv주소

*패딩 : 6*6행렬의 테두리를 한칸씩 늘려 0으로 채움 -> 이미지로부터 특징을 찾아내는 과정에서 원본 이미지의 크기 줄어드는것 방지
-> Valid : 패딩을 안한다
-> Same : 0으로 테두리 채움 -> 연산 결과도 인풋 이미지랑 동일하게 나옴
-> stride = 2 로 설정하면 두칸씩 이동해서 연산 수행

*Pooling(특정 영역을 대표하는 대표값 추출하는 행위)
-Max pooling : pool size를 (2,2)로 주면 연산 수행된 결과에서 2*2크기로 겹치지 않게 나누고 그 영역에서 가장 큰값 뽑아냄
-Average pooling : 평균값 뽑아냄

*Drop out : 학습할때마다 batch size단위로 렌덤하게 특징들을 선택하여 다각도의 측면에서 모델 만ㅁ들어짐 
"""
```

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
```
```python
df = pd.read_csv("data3/house_train.csv")
df=pd.get_dummies(df)
df=df.fillna(df.mean())
```
-  열이 증가함 ->각각의 열에 해당하는 값들이 종류로 들어가있다 판단시 더미화 되서 원핫 인코딩 됨

```python
df_corr=df.corr()
df_corr_sort=df_corr.sort_values('SalePrice', ascending=False)
```
- 상관계수를 출력한 값을 저장하고 그 값을 내림차순으로 정렬했다.

```python
cols=['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']
sns.pairplot(df[cols])
plt.show()
```
- 6개의 열을 지정하고 그 열들의 pairplot을 출력했다.

```python
cols_train=['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']  #집값 결정하는 독립변수 5개 저장 
X_train_pre = df[cols_train]
y = df['SalePrice'].values
X_train, X_test, y_train, y_test = train_test_split(X_train_pre, y, test_size=0.2)
```
- 집값을 결정하는 변수 5개를 저장하고 y에는 가격을 저장했다. 그리고 그 값들을 train_test_split을 이용해서 분리시켰다 (비율 8:2)

```python
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))  #input_dim에 X_train의 차원 
model.add(Dense(30, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(1))
model.summary() 
```
- Dense 높아지는것 : 모델이 데이터의 다양한 특징을 학습하도록 도울 수 있음
    - 노드 수를 감소시키는 경우에는 모델의 용량을 줄이고, 더 간단한 모델을 만들 수 있으며, 이로 인해 과소적합의 위험을 줄일 수 있다 
    - 그러나 이것이 항상 필요한 것은 아니며, 작업에 따라 높은 노드 수가 필요할 수도 있음
- 회귀 작업에서는 출력이 실수값인 경우가 일반적, activation 함수를 사용하지 않으면 모델은 임의의 실수값을 예측할 수 있음
- 분류(Classification) 작업의 경우, 일반적으로 출력 레이어에는 적절한 activation 함수를 사용해야 합니다
- Dense 레이어의 activation 함수 여부는 모델의 작업 유형에 따라 다르며, 회귀 작업에는 일반적으로 사용하지 않고, 
분류 작업에는 사용하는 것이 좋습니다.
```python
model.compile(optimizer ='adam', loss = 'mean_squared_error')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
modelpath="./data/model/house_best.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)
history = model.fit(X_train, y_train, validation_split=0.25, epochs=2023, batch_size=32, callbacks=[early_stopping_callback, checkpointer])
```
- 여기까지가 모델을 실행하는 코드다.
    - loss를 accuracy로 주면 분류 문제
    - mean_squared_error은 연속형 값을 비교할때 사용해 모델이 좋아졌는지 나빠졌는지 비교가능 

```python
model.predict(X_test)

real_prices =[]
pred_prices = []
X_num = []


n_iter = 0
Y_prediction = model.predict(X_test).flatten()
for i in range(30):
    real = y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.2f}, 예상가격: {:.2f}".format(real, prediction))
    real_prices.append(real)
    pred_prices.append(prediction)
    n_iter = n_iter + 1
    X_num.append(n_iter)
실제가격: 134000.00, 예상가격: 146997.64
실제가격: 555000.00, 예상가격: 350596.16
실제가격: 348000.00, 예상가격: 288766.91
실제가격: 196000.00, 예상가격: 232310.58
실제가격: 110000.00, 예상가격: 112227.27
실제가격: 295493.00, 예상가격: 252427.58
실제가격: 190000.00, 예상가격: 206682.89
실제가격: 137500.00, 예상가격: 177517.55
실제가격: 82500.00, 예상가격: 141089.09
실제가격: 302000.00, 예상가격: 305604.50
실제가격: 174900.00, 예상가격: 167803.77
실제가격: 89471.00, 예상가격: 171337.73
실제가격: 156000.00, 예상가격: 150309.84
실제가격: 125000.00, 예상가격: 230447.69
실제가격: 118000.00, 예상가격: 129567.74
실제가격: 287000.00, 예상가격: 268125.22
실제가격: 112000.00, 예상가격: 164520.67
실제가격: 162000.00, 예상가격: 166746.80
실제가격: 165000.00, 예상가격: 160160.52
실제가격: 266000.00, 예상가격: 232125.36
실제가격: 146500.00, 예상가격: 161835.17
실제가격: 164500.00, 예상가격: 185196.78
실제가격: 180500.00, 예상가격: 199337.55
실제가격: 143000.00, 예상가격: 188920.98
실제가격: 177000.00, 예상가격: 169067.73
실제가격: 172500.00, 예상가격: 206069.47
실제가격: 153000.00, 예상가격: 179020.02
실제가격: 206900.00, 예상가격: 224418.39
실제가격: 173000.00, 예상가격: 188908.25
실제가격: 144500.00, 예상가격: 181900.20
```

```python
plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
plt.legend()
plt.show()
```
- 선 그래프로 시각화했다.
## CNN
```python
from tensorflow.keras.datasets import mnist  # 숫자 데이터셋
from tensorflow.keras.utils import to_categorical # 데이터타입 카테고리로 변환
import matplotlib.pyplot as plt
import sys
```
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("학습셋 이미지 수 : %d 개" % (X_train.shape[0]))
print("테스트셋 이미지 수 : %d 개" % (X_test.shape[0]))
```
- 학습셋 이미지 수 : 60000 개
테스트셋 이미지 수 : 10000 개

```python
plt.imshow(X_train[0])   #5
y_train[0]  #정답 5
plt.imshow(X_train[1], cmap='Greys')
y_train[1]
```
```python
for x in X_train[0]:
    for i in x:
        sys.stdout.write("%-3s" % i)
    sys.stdout.write('\n')
```
- mnist의 데이터셋을 그림으로 확인하고 그 그림을 sys를 이용해서 그 그림의 픽셀당 크기를 확인했다.

## 일반적인 DNN(ANN, 딥러닝) 구조
```python
X_train = X_train.reshape(X_train.shape[0], 784).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 784).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```
- 픽셀 한칸에 들어갈 수 있는 데이터의 크기는 255가 최대이니 255로 나눠 정규화를 해주고, 정답칸에는 카테고리화를 해 0~9까지의 수를 0000000001 와 같은 식으로 표현하는 코드를 만들었다.
    - 원핫인코딩과 같음

```python
model = Sequential()
model.add(Dense(860 ,input_dim = 784, activation='relu'))
model.add(Dense(10 , activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
MODEL_DIR = './data/model/'
modelpath="./data/model/MNIST_MLP.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])
```
- 모델을 만들고 실행하는 코드다.

```python
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
313/313 [==============================] - 1s 2ms/step - loss: 0.1489 - accuracy: 0.9777

 Test Accuracy: 0.9777
```

-  정답률은 0.9777이 나왔다.
## CNN구조
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
```
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```
- shape[0]으로 그림의 수를 파악시켜 reshape하고 실수타입으로 변경한 뒤, DNN과 마찬가지로 정규화를 했다.
    - 정답들은 categorical을 이용해서 카테고리화 시켰다.

```python
model = Sequential()
#Conv2d(32는 필터의 갯수, 필터 크기, 입력 이미지 크기, 활설화 함수
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))  
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))   # 25% 만큼이 제외되고 75%만 훈련에 참가한다
model.add(Flatten())      # Flatten 한다음에 Dense가 나옴 
model.add(Dense(128,  activation='relu'))
model.add(Dropout(0.5)) # 절반 떨굼
model.add(Dense(10, activation='softmax'))
```
- CNN의 모델 실행 코드다 주의를 해야할것은 Conv2D부분이다. 이미지크기를 잘 주어야 하고 필터의 크기와 갯수가 중요하다.

```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
modelpath="./data/model/MNIST_CNN.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])
```
```python
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
313/313 [==============================] - 2s 4ms/step - loss: 0.0345 - accuracy: 0.9918

 Test Accuracy: 0.9918
```
- 같은 데이터로 학습을 하고 정확도를 비교했는데 DNN에 비해서 CNN의 정확도가 0.02정도 높게 측정됐다.

## DNN

```python
from keras.datasets import fashion_mnist
# load dataset
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
X_train = trainX.reshape(trainX.shape[0], 784).astype('float32') / 255
X_test = testX.reshape(testX.shape[0], 784).astype('float32') / 255
y_train = to_categorical(trainy,10)
y_test = to_categorical(testy,10)
```
- 데이터 전처리의 과정이다.

```python
model=Sequential()
model.add(Dense(500, input_dim = 784, activation='relu'))
model.add(Dense(10 , activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
MODEL_DIR = './data/model/'
modelpath="./data/model/MNIST_MLP.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
history=model.fit(X_train,y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
313/313 [==============================] - 0s 1ms/step - loss: 0.3518 - accuracy: 0.8868

 Test Accuracy: 0.8868
```

#CNN

```python
from keras.datasets import fashion_mnist
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()
X_train=trainX.reshape(trainX.shape[0],28,28,1).astype('float32') / 255
X_test=testX.reshape(testX.shape[0],28,28,1).astype('float32') / 255
y_train=to_categorical(trainy)
y_test=to_categorical(testy)
model=Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))  
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,  activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
modelpath="./data/model/MNIST_CNN.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, validation_split=0.25, epochs=30, batch_size=200, verbose=0, callbacks=[early_stopping_callback,checkpointer])
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
313/313 [==============================] - 2s 5ms/step - loss: 0.2562 - accuracy: 0.9288

 Test Accuracy: 0.9288
```
- 복습겸 다른 데이터셋으로 두번 더 연습했다.