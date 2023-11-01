# 복습

```python
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense  # 모델에 층(layer)를 추가하기 위한 라이브러리
import numpy as np
import pandas as pd
```
```python
Data_set = np.loadtxt("data_/ThoraricSurgery3.csv", delimiter=",")
X = Data_set[:,0:16]
y = Data_set[:,16]
```
- 데이터를 불러온 뒤 정답열을 y로 저장, 그 외의 값은 X로 저장했다.

```python
model = Sequential()
model.add(Dense(30, input_dim = 16, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.fit(X, y, epochs = 10, batch_size = 5)
```
- 모델을 만들고 중간에 hidden layer를 만들고 난 뒤 세부옵션을 설정했다. 그 후에 실행.
    - 만약 모델을 다른 값으로 다시 실행시키고 싶은 경우는 처음부터 다시 시작해야한다.
        - batch_size 의 값이 높으면 더 빨라지지만, 수치가 너무 높아진다면 메모리상에 무리가 가고, 학습이 잘 안된다.
## 예측모델
```python
x = np.array([2, 4, 6, 8])
y = np.array([81, 93, 91, 97])
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값:", mx) # x의 평균값: 5.0
print("y의 평균값:", my) # y의 평균값: 90.5
```
- 각 값의 평균값을 구했다.
```python
divisor = sum([(i - mx)**2 for i in x])
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d
dividend = top(x, mx, y, my)
print("분모:", divisor)
print("분자:", dividend)
```
- 미분을 하기 위해 각 값을 구했다.
    - 분자에는 x의 편차와 y의 편차의 곱의 합, 그리고 분모에는 x편차의 제곱합이 들어간다.
```python
a = dividend / divisor # 기울기
b = my - (mx*a) # y절편
print("기울기 a =", a)
print("y절편 b =", b)
```
- 기울기와 y절편을 구하는 공식이다.
```python
def predict(x):
    return a * x + b # 회귀모델 = 2.3 * X + 79
predict_result = []
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부시간 = %.f, 실제점수=%.f, 예측점수=%.f" % (x[i], y[i], predict(x[i])))
```
- 미리 예측값을 구하는 함수를 정의하고 함수 안에 값을 넣어 점수를 예측하는 코드를 구현했다.

```python
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.show()
```
- 시각화를 했다.

```python
a = 0
b = 0
lr = 0.03
epochs = 2001
```
- 기울기 a와 절편 b의 값을 초기화하고, 학습률을 정했다.
    - epochs는 몇 번 반복될지를 설정한것이다.

```python
for i in range(epochs):   
    y_pred = a * x + b 
    error = y - y_pred
    a_diff = (2/n) * sum(-x * (error))
    b_diff = (2/n) * sum(-(error)) 
    a = a - lr * a_diff 
    b = b - lr * b_diff
    if i % 100 == 0:
        print("epoch=%.f, 기울기=%.04f, 절편=%.04f" % (i, a, b))
```
- 경사하강법의 코드다.
    - 먼저 epochs의 수 만큼 반복, 예측값을 구하는 식과 실제 값과 비교한 오차를 error로 저장하는 식이다.
    - 오차함수를 각각 a와 b로 편미분한 값이다.
    - 학습률을 곱해 기존의 a값과 b값을 업데이트했다.
    - 100번 반복될 때마다 현재의 a와 b의 값을 출력했다.

```python
epoch=0, 기울기=27.8400, 절편=5.4300
epoch=100, 기울기=7.0739, 절편=50.5117
epoch=200, 기울기=4.0960, 절편=68.2822
epoch=300, 기울기=2.9757, 절편=74.9678
epoch=400, 기울기=2.5542, 절편=77.4830
epoch=500, 기울기=2.3956, 절편=78.4293
epoch=600, 기울기=2.3360, 절편=78.7853
epoch=700, 기울기=2.3135, 절편=78.9192
epoch=800, 기울기=2.3051, 절편=78.9696
epoch=900, 기울기=2.3019, 절편=78.9886
epoch=1000, 기울기=2.3007, 절편=78.9957
epoch=1100, 기울기=2.3003, 절편=78.9984
epoch=1200, 기울기=2.3001, 절편=78.9994
epoch=1300, 기울기=2.3000, 절편=78.9998
epoch=1400, 기울기=2.3000, 절편=78.9999
epoch=1500, 기울기=2.3000, 절편=79.0000
epoch=1600, 기울기=2.3000, 절편=79.0000
epoch=1700, 기울기=2.3000, 절편=79.0000
epoch=1800, 기울기=2.3000, 절편=79.0000
epoch=1900, 기울기=2.3000, 절편=79.0000
epoch=2000, 기울기=2.3000, 절편=79.0000
```
```python
hx = a * x + b #모델의 예측값
```
- array([83.59999984, 88.19999992, 92.8       , 97.40000008])
    - array([81, 93, 91, 97]) 이건 실제 정답이다.

## 로지스틱 회귀 모델
```python
x = np.array([2, 4, 6, 8, 10, 12, 14])
y = np.array([0, 0, 0, 1, 1, 1, 1])
model = Sequential()
model.add(Dense(1, input_dim = 1, activation = "sigmoid"))
model.compile(optimizer='sgd' ,loss='binary_crossentropy') # binary_crossentropy = 2진분류
```
- 리스트에 값을 넣고 바로 모델을 만들었다.

```python
model.fit(x,y, epochs = 5000)
if model.predict([7])[0][0]*100 > 50:
    print("합격으로 예상")
else:
    print("불합격으로 예상")
```
- 1/1 [==============================] - 0s 30ms/step 출력값이다.

```python
# 다층 퍼셉트론
# 퍼셉트론이란 입력값을 입력 받아 처리를 해서 결과를 도출해내는 역할을 담당
# 신경망 상에서 뉴런의 역할을 하는것이 퍼셉트론
# 뉴런의 역할? 연산을 하고 다음 뉴런한테 전달하는 것
```
```python
# 가중치와 바이어스
w11 = np.array([-2, -2])
w12 = np.array([2, 2])
w2 = np.array([1, 1])
b1 = 3
b2 = -1
b3 = -1
# 퍼셉트론
def MLP(x, w, b):
    y = np.sum(w * x) + b
    if y <= 0:
        return 0
    else:
        return 1
def NAND(x1,x2):
    return MLP(np.array([x1, x2]), w11, b1)
def XOR(x1,x2):
    return AND(NAND(x1, x2),OR(x1,x2))
def AND(x1,x2):
    return MLP(np.array([x1, x2]), w2, b3)
def OR(x1,x2):
    return MLP(np.array([x1, x2]), w12, b2)
for x in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    #print(x)
    #y = OR(x[0], x[1])
    #y = AND(x[0], x[1])
    y = XOR(x[0], x[1])
    print(y)
```
- OR, AND, XOR을 구하는 함수다. 기본 2중함수 구조다.
```python
df = pd.read_csv("data_/pima-indians-diabetes3.csv")
colormap = plt.cm.gist_heat   
plt.figure(figsize=(12,12))  
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=colormap, linecolor='white', annot=True)
plt.show()
```
- 당뇨병에 관한 데이터의 상관관계지표를 시각화했다.

```python
plt.hist(x=[df.plasma[df.diabetes==0], 
            df.plasma[df.diabetes==1]], bins=30, histtype='barstacked', label=['normal','diabetes'])
plt.legend()
plt.hist(x=[df.bmi[df.diabetes==0], 
            df.bmi[df.diabetes==1]], bins=30, histtype='barstacked', label=['normal','diabetes'])
plt.legend()
```
- 각각 plasma열과 bmi열이 당뇨병의 발병과 어떤 연관이 있는지 시각화했다.

```python
X = df.iloc[:,0:8]
y = df.iloc[:,8]
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', name='Dense_1'))
model.add(Dense(8, activation='relu', name='Dense_2'))
model.add(Dense(1, activation='sigmoid',name='Dense_3'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=10)
```
- 딥러닝의 코드다. 기본적으로 지금까지 한 것의 반복작업이다.

```python
df = pd.read_csv('data_/sonar3.csv', header = None)
X = df.iloc[:,0:60]
y = df.iloc[:,60]
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X, y, epochs=200, batch_size=10)
```
- 데이터를 불러와서 다른 분류작업 없이 바로 모델을 만들고 실행시켰다.

```python
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
```
- 여기서 shuffle값을 true로 준 이유는 전체데이터가 1 : 70%, 0:30% 비율이라면 1과 0이 train(7:3), test(7:3)의 비율로 나눠진다.
```python
model = Sequential()
model.add(Dense(24,  input_dim=60, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=model.fit(X_train, y_train, epochs=200, batch_size=10)
score = model.evaluate(X_test, y_test)
score[1]
model.save("myModel.hdf5")
```
- 지금까지의 모델을 만들고 저장하는 과정이다.

```python
del model
myModel=load_model("myModel.hdf5")
myModel.evaluate(X_test, y_test)
```
- 마찬가지로 모델을 지우고 그 모델을 새로이 불러와 작동이 가능한지를 시험해봤다.
```python
df = pd.read_csv('data_/wine.csv', header=None)
X = df.iloc[:,0:12]
y = df.iloc[:,12]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=50, batch_size=500, validation_split=0.25) # 0.8 x 0.25 = 0.2
model.evaluate(X_test, y_test)
```
- 모델을 만들고 train과 test를 나눈 데이터에서 train을 또 train과 validation으로 나눴다.
```python

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
modelpath="./data/model/all/{epoch:02d}-{val_accuracy:.4f}.hdf5"
```
- 모델을 만들고 모델의 경로를 지정했다.

```python
from tensorflow.keras.callbacks import ModelCheckpoint
```
- 모델 체크포인트 라는 클래스를 이용해서 모델 저장이 가능하다
```python
checkpointer = ModelCheckpoint(filepath=modelpath, verbose=1)
history=model.fit(X_train, y_train, epochs=50, batch_size=500, validation_split=0.25, verbose=0, callbacks=[checkpointer])
```
- verbose는 출력 보기 형식을 지정한다.
```python
score=model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, verbose=0, validation_split=0.25)
hist_df=pd.DataFrame(history.history)
```
- 모델을 만들고 수행한 뒤 history.history로 dataframe을 만들었다.

```python
y_vloss=hist_df['val_loss']
y_loss=hist_df['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, "o", c="red", markersize=2, label='Testset_loss')
plt.plot(x_len, y_loss, "o", c="blue", markersize=2, label='Trainset_loss')

plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
```
- 시각화 작업이다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import pandas as pd

# 데이터를 입력합니다.
df = pd.read_csv('data_/wine.csv', header=None)

# 와인의 속성을 X로 와인의 분류를 y로 저장합니다.
X = df.iloc[:,0:12]
y = df.iloc[:,12]

#학습셋과 테스트셋으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 모델 구조를 설정합니다.
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#모델을 컴파일합니다.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```
- 똑같은 작업이다. 데이터를 불러와서 정답열과 분리, train과 test로 나누고 모델 구조 설정 후 컴파일이다.

```python
df = pd.read_csv("wdbc.data", header = None)
# B(양) : 0,  M(악) : 1

# 트레인 : 0~450번 데이터
# 테스트 : 451~ 568번까지 데이터

# 딥러닝 모델 생성 -> 정확도 향상
```
- 오늘의 과제다.

```python
df[1][df[1] == "B"] = 0
df[1][df[1] == "M"] = 1
data = df.drop(1,axis = 1)
numeric_data = df.iloc[:, :]
```
- 정답열을 먼저 수치화하고 그 열을 분리해 저장했다.
    - numeric_data에 df를 저장했다..

```python
scaled_df = pd.DataFrame(standardized_data, columns=numeric_data.columns)
X_train = scaled_df[:451]
y_train = df[1][:451]
X_test = scaled_df[451:]
y_test = df[1][451:]
```
- 직접 train과  test를 나눴다.

```python
model = Sequential()
model.add(Dense(30, input_dim = 31, activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))
model. summary()

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
y_train = y_train.astype(int)
y_test = y_test.astype(int)
```
- 모델을 생성하고 구조를 설정하는 과정이다.
    - 정답열을 int타입으로 변경하였다.

```python
# 학습이 언제 자동 중단 될지를 설정합니다.
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)

#최적화 모델이 저장될 폴더와 모델의 이름을 정합니다.
modelpath="./data2/model/Ch14-4-bestmodel.hdf5"

# 최적화 모델을 업데이트하고 저장합니다.
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True)

#모델을 실행합니다.
history=model.fit(X_train, y_train, epochs=2000, batch_size=500, validation_split=0.25, verbose=1, callbacks=[early_stopping_callback,checkpointer])
```
- 모델을 실행시켰다.