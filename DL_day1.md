# 인공신경망

```python
# 신경망이란
"""
인간의 뉴런작용과정에서 아이디어를 착안, 시냅스를 통해 다른 뉴런으로부터 전달받고
전달하는 역할
신경망이란 자극이 전달되는 과정을 표현한것
레이어와 퍼셉트론으로 뉴런을 표현
인간신경망의 뉴런 =퍼셉트론, ..뉴런과 뉴런사이를 이으며 자극전달하는
시냅스는 여러개의 레이어를 잇는 가중치...
                                        activation fuction:시그노이드,엘루,스텝,리니어등
                                        활성화함수(연산결과를 다음뉴런으로 전달여부 판단)
                                        전달시 어떤값으로 전달할지를 판단
inputs    ---------->     hiddn layer     ---------활성화함수------------>    output
입력신호    웨이트        뉴런(사각모양)
            (가중치)        퍼셉트론                                         1 if 조건>0
X1~Xn까지    W1~Wn          가중평균                                         -1 otherwise
                            시그마wi*Xi
두개의 입력변수 x1,x2가 존재할 경우, w1.x1+w2.x2+w0 =y  , 다중선형회귀와 유사
한개의 변수가 있을경우,  wx+b= y
가령, w1.x1+w2.x2+b =0 인 직선의 방정식으로 분류시..
직선을 기준으로 왼쪽은 동그라미 오른쪽은 세모일 경우
임의의 점이 직선 왼쪽에 입력시,직선의 방정식에서 w1과 w2가 정해진 상황에서,
x1,x2가 입력된 임의의 점은,,동그라미와 세모중 무엇인가?
0보다 크면 동그라미, 작으면 삼각형
위 직선과 같은 모델을 만드는 것 ==> 로지스틱 리그레션,svm, 의사결정트리등이
바로 이러한 선을 찾아내는 모델을 의미함...분류가 과도할 경우 오버피팅의 문제발생
신경망은...w1.x1+w2.x2+b 을 activation fuction에 대입후,  output 도출
                            Ex) 시그모이드
## 유용한 사이트 --> 강좌 적극추천 http://cs231n.stanford.edu/
   향후 혼자공부할때 수업을 꼭 들어보기를 추천/ 이미지분류에 특화된 강좌
* 소프트맥스의 경우, 다중분류기에 추출한 값이 softmax로 들어감--> 확률로 바뀜
  각각의 분류기로부터 추출한 값을 확률로 변환해주는 함수,,,소프트맥스...
  이 구조에서, hidden layer가 추가 된 경우를 "신경망"이라고 함
 가령, x, y를 학습시켜 모델을 만들려고 함(지도학습) ==> linear regression, logistic regression
 , svm, random forest, decision tree, knn등 다양한 알고리즘 있음.
 x라는 변수하나일 경우, 혹은 여러개일 경우로 학습하는 경우==> 비지도학습==> kmean, pca, auto-clustering등
 신경망은 "지도학습"에 해당됨!!!
             얇은 신경망                vs             깊은 신경망
input layer--> hidden layer  output         input layer--> hidden layer  output
                                                          hidden layer
만약 x1,x2 2개의 독립변수로 구성된 데이터셋이 있다면,,or연산에 해당
이 데이터를 바탕으로 학습을 하고, 모델을 구축하려고 함. 여기서는 결국 y값을 예측값으로
만들려는 방식, 0또는 1로 분류하려고 함..좌표평면상 x1, x2평면상 표시하면..
x1,x2에 어떤값이 나오도라도 0또는 1로 분류하려면..데이터를 잘 분리하는 직선을 찾아야함
직선을 기준으로 1과 0으로 구분....이 직선의 방정식은  Y=w1.x1+w2.x2+b, 이런형식
따라서, 분류기 기준값보다 크냐 작느냐에 따라 0과 1로 분류하게 됨..
만약 이런분류기가 여러개 있는 경우를 "다중분류"라고 함, 만약 변수가x1,x2가 있는데,
y값이 A,C,C,B와 같은 형식으로 되어 있다면, 이데이터를 가지고 모델을 만들려고할때,
Y종류가 3가지(A, B, C중의 하나)일 경우, 직선하나로는 A,B,C를 분류하는데 한계존재
그래서, A인지 아닌지 분류하는 직선, B인지 아닌지, C인지 아닌지를 분류하는 직선등 3개의 직선이
필요... 가령 어떤점이 B인지 아닌지,A인지 아닌지 분류선에 포함될 경우, 거리를 구해
어느선에 더 가까운지를 기준으로 확률적으로 평가해 결과값을 도출....
이렇게 분류기를 만드는데, 신경망에서는 중간에 hidden layer가 들어가고, 특징을 찾는 다양한
함수들이 들어감
요컨데 기본적 분류모델의 작동방식
                                            시그노이드(이진)
입력데이터 --> 분류기(이진 또는 다중분류)--> 소프트맥스(다중) --> 0~1의 확률값으로 나타냄
               Y=w1.x1+w2.x2+b
신경망의 경우...
입력데이터 --> 분류기(이진 또는 다중분류)--> 히든레이어  ---함수적용--->   시그노이드(이진
x1,x2,x3       w1,w2,w3                     노드/뉴런                      소프트맥스(다중) --> 0~1의 확률값으로 나타냄
                                  각각의 히든레이어의 모든 노드에 연결
                                  입력차원3, 노드4차원 ==> 총13개 노드(b포함)
중간에 히든레이어가 포함되어 있음. 나머지 부분/방식은 거의 동일함...
결국 히든 레이어와 노드의 수를 지정하는것이 중요
신경망을 공부하는 목적은 회귀와 분류를 하기 위함
경사하강법으로 신경망이 학습이 된다.
딥러닝이라고 하는것은 히든레이어를 2개 이상으로 두는 것을 말한다.
히든레이어는 추상화된 특징
히든레이어를 추가하기 좋은 경우
"""
```
```python
"""
케라스(딥러닝 프레임웍)을 이용한 점수 예측
입력 특징 : 공부시간, 출석일수, 과제 제출 횟수
출력 : 시험점수

???프레임웍 : framework(구조화 된 틀)
쉽고 적은 비용으로 개발 할 수 있도록 해주는 library들의 집합체
텐서플로우(케라스), 파이토치, 까페
"""
```

- 오늘 배운 신경망과 깊은 신경망에 대한 이론이다.

```python
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([layers.Dense(units=1, input_shape=[3])])
```
- 사용한 모듈들과 신경망 모델을 생성하고 정의하는 코드다.   
- units = 출력의 수, input_shape = 입력 차원를 의미한다.

```python
model = keras.Sequential([
    layers.Dense(units=4, input_shape=[2], activation="relu"),
    layers.Dense(units=3, activation="relu"),
    layers.Dense(units=1, activation="sigmoid")    
])
```
- 이 코드의 의미는 hidden layer를 각각 4개와 3개 그리고 output으로 1개를 출력하며 최초의 입력값은 2개를 가진다.
    - 또, 마지막 출력값은 분류모델이라면 sigmoid를 사용한다.

```python
model.summary()
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_1 (Dense)             (None, 4)                 12        
                                                                 
 dense_2 (Dense)             (None, 3)                 15        
                                                                 
 dense_3 (Dense)             (None, 1)                 4         
                                                                 
=================================================================
Total params: 31 (124.00 Byte)
Trainable params: 31 (124.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
- 코드 아래의 값이 출력값이다.
```python
mldel.compile(optimizer="adam", loss="mae")
```
- optimizer는 최적의 함수를 찾는 값으로 무엇을 줄건지를 의미하고 loss는 손실값을 어떤 값으로 표현할것인지를 의미한다.

```python
import pandas as pd
red_wine = pd.read_csv("red-wine.csv")
```
- pandas를 import하고 가지고 있는 wine 데이터를 red_wine으로 저장했다.
```python
df_train = red_wine.sample(frac = 0.7, random_state = 20231031)
df_test = red_wine.drop(df_train.index)
len(df_train) # 1119
len(df_test) # 480
```
- sample 함수의 frac값을 이용해서 train과  test로 각각 데이터를 나눠 저장했다.

```python
max_ = df_train.max()
min_ = df_train.min()
df_train = (df_train - min_) / (max_ - min_)
max_ = df_test.max()
min_ = df_test.min()
df_test = (df_test - min_) / (max_ - min_)
# 정규화 과정
```
- 최대값과 최소값을 정의하고 공식을 직접 입력해 정규화 시켰다.

```python
df_train.drop("quality", axis = 1, inplace = True)
df_test.drop("quality", axis = 1, inplace = True)
y_train = red_wind.quality.drop(df_test.index)
y_test = red_wine.quality.drop(df_train.index)
```
- quality열은 종속변수로 정답열인데 이것 역시 정규화가 되어서 열을 drop시키고 그 값을 따로 저장해줬다.

```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
pd.get_dummies(red_wine.quality)
y_test_ohe = ohe.fit_transform([y_test])
```
- 원 핫 인코딩의 과정이다. get_dummies의 방법도 있지만 이 코드역시 인코딩이 가능하다.
```python
model=keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[11]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(11, activation='softmax')    
])
model.compile(
    optimizer='adam',
    loss='mae',
)
model.fit(
    df_train, y_train, epochs=10
)
Epoch 1/10
35/35 [==============================] - 1s 5ms/step - loss: 5.5257
Epoch 2/10
35/35 [==============================] - 0s 5ms/step - loss: 5.5257
Epoch 3/10
35/35 [==============================] - 0s 5ms/step - loss: 5.5257
Epoch 4/10
35/35 [==============================] - 0s 5ms/step - loss: 5.5257
Epoch 5/10
35/35 [==============================] - 0s 5ms/step - loss: 5.5257
Epoch 6/10
35/35 [==============================] - 0s 5ms/step - loss: 5.5257
Epoch 7/10
35/35 [==============================] - 0s 5ms/step - loss: 5.5257
Epoch 8/10
35/35 [==============================] - 0s 5ms/step - loss: 5.5257
Epoch 9/10
35/35 [==============================] - 0s 5ms/step - loss: 5.5257
Epoch 10/10
35/35 [==============================] - 0s 5ms/step - loss: 5.5257
```
- 전과 같이 모델을 훈련시키는 과정이다.

```python
model.predict(y_test_ohe)
15/15 [==============================] - 0s 1ms/step
array([[0.08463182, 0.08999782, 0.08647078, ..., 0.09339071, 0.09004647,
        0.09355006],
       [0.08430382, 0.08962122, 0.08710458, ..., 0.09412811, 0.09214524,
        0.09204488],
       [0.08553871, 0.08983689, 0.08797844, ..., 0.09316944, 0.09112994,
        0.09203072],
       ...,
       [0.08529732, 0.0886083 , 0.08785136, ..., 0.09329405, 0.09080723,
        0.09286953],
       [0.08669287, 0.08776188, 0.08784359, ..., 0.09380012, 0.09033063,
        0.09384383],
       [0.08485531, 0.08794297, 0.08805116, ..., 0.09354819, 0.09085368,
        0.09362008]], dtype=float32)

```
- 와인의 정답을 예측하는 코드다.
```python
df = pd.read_csv("diabetes.csv")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```
- 과거에 다뤄보았던 당뇨병 데이터를 가지고 왔다.

```python
x = df.iloc[:, 0:8]
y = df.iloc[:, 8]
model = Sequential()
model.add(Dense(12, input_dim=8, activation = "relu"))
model.add(Dense(8, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))
```
- x와 y 를 데이터와 정답열로 정의해주고 모델을 만들었다.
    - 코드는 다르지만 한번에 모델을 만드는것이 아니라 .add를 이용해서 이렇게 만드는 방식 역시 존재한다.
```python
model.summary()
Model: "sequential_5"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_16 (Dense)            (None, 12)                108       
                                                                 
 dense_17 (Dense)            (None, 8)                 104       
                                                                 
 dense_18 (Dense)            (None, 1)                 9         
                                                                 
=================================================================
Total params: 221 (884.00 Byte)
Trainable params: 221 (884.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

```python
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(x,y, epochs = 10, batch_size = 5)
```
- epochs는 데이터를 몇번 학습을 시킬것인지를 정하는 값이고, batch_size는 한번에 몇개의 데이터를 넣을 것인지를 정하는 값이다.

```python
df = pd.read_csv('iris3.csv')
X = df.iloc[:,0:4]
y = df.iloc[:,4]
y = pd.get_dummies(y)
```
- 정답열만 따로 저장해주고 그 정답열을 get_dummies로 원핫인코딩을 해줬다.

```python
model = Sequential()
model.add(Dense(16,  input_dim=4, activation='relu'))
model.add(Dense(8,  activation='relu'))
model.add(Dense(3,  activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y,epochs=40, batch_size=5)
```
- 마지막 코드다. 모델을 만들고 16,8의 hidden layers, 그리고 3개의 출력값, 최초의 4개의 입력값을 각각 저장해줬다.
    - 원 핫 인코딩이라 마무리는 softmax를 적어줬다.
        - 다중 클래스 분류시 사용하는 categorical_crossentropy를 적어주고 최적화 기법은 역시나 adam, 척도는 정확도의 값을 주었다.
    - 40번을 반복하게 만들었고 한번에 5개의 데이터가 들어가 학습을 하게 설정했다.