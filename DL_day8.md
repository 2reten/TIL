# 워드 임베딩,LSTM과 RNN

```python
지도 학습의 훈련 데이터에는 레이블이 반드시 있어야 한다.
# 데이터 전처리(준비) 과정
# 제목 열 추출
# 공백 분리/ 형태소 분리...
# 불용어 제거, 조사..., 정규표현식
# 유일한 단어 -> 길이 (코퍼스 크기)
# 각 단어 숫자 부여
# 가장 긴 문장 길이 파악
# 모든 문장 길이를 동일하게 padding
```
```python
# 데이터 전처리 과정이 끝난 후 
# LSTM/RNN 설계
# timestemp
# word dim - 만약 차원수가 크다면 embedding을 통해서 차원을 축소해야한다.
# batch-size
# LSTM 셀 추력 차원
```
```python
# 패스트 텍스트
# 단어를 벡터로 만드는 다른 방법이다. word2vec이후에 나와 확장판이라고 생각하면 된다.
# word2vec보다 fasttext가 더 성능이 우세하다.
# word2vec은 단어를 쪼개질 수 없는 단위로 생각한다면, fasttext는 하나의 단어 안에도 여러 단어들이 존재하는 것으로 간주한다(subword)
# 이를 고려하여 학습한다.
# 글자 단위 n-gram의 구성으로 취급 n을 1로 준다면 apple의 경우 <,a,p,p,l,e,>이고  2로 준다면 <a,ap,pp,pl,le,e>와 같은 형식이다.
# 내부단어를 분리해 토큰화 할 때 시작과 끝에 각각 <,>를 도입시켜 단어의 시작과 끝을 알린다.
# 2-gram을 기준으로 생각하면 <a,ap,pp,pl,le,e>,<apple>로 총 7개의 토큰이 생긴다.
# 한국어의 경우 오타를 찾아내기에 좋다.
# n-gram은 최소값과 최대값을 지정해야한다 default는 3,6이다.
# 벡터화는 word2vec을 수행해 벡터화한다.
# 벡터들을 모두 더해 <apple>이 아닌 apple의 값에 저장한다.
# fasttext의 인공 신경망을 학습한 후에는 데이터 셋의 모든 단어의 각 n-gram에 대해서 워드 임베딩이 된다.
# -> 내부 단어들을 통해서 모르는 단어와의 유사도를 계산이 가능해진다.
# word2vec에는 등장 빈도수가 적었던 단어에 대해서는 임베딩의 정확도가 높지 않다는 단점이 있다.
# fasttext는 단어가 희귀단어라고 하더라도, 단어의 n-gram이 다른 단어의 n-gram과 겹치는 경우라면 비교적 높은 임베딩 벡터값을 가진다.
# 오타의 경우는 word2vec에서 희귀단어가 되어 임베딩이 제대로 되지 않지만 fasttext는 이에 대해서도 일정 수준의 성능을 보인다.
# fasttext도 기본적인 emdbedding방식은 word2vec과 유사하다.
# fasttext 한국어의 경우
# "자연어처리"를 나누면 <자연, 자연어, 연어처,어처리, 처리>로 나누어지고, 한국어의 경우는 자모 단위로 또 분리가 가능하다.
# "자연어처리" => ㅈ ㅏ _ ㅇ ㅕ ㄴ ㅇ ㅓ _ ㅊ ㅓ _ ㄹ ㅣ _ -> 이 값을 벡터화 시켜 유사도를 측정한다.
```
```python
# BERT
# 엘모가 양방향 모델 두개를 결합하는 방식이라면 BERT는 단일 모델로 양방향을 모두 학습하는 방식이다.
# BERT가 더 빠르다.
```

```python
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import reuters       # 로이터 뉴스 데이터셋 불러오기
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
```
- 오전 실습에 사용한 모듈들이다.
```python
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)
```
- num_words는 뉴스 기사에서 가장 많이 등장하는 상위 1000개의 단어만 가져온것이다.  => 로이터 뉴스 8982건에 대해 가장 많이 언급된 1000개 단어들로 구성된 상태다.
```python
X_train.shape # (8982, ) 뉴스 기사
X_train[0]
np.max(y_train) # 45
np.min(y_train) # 0
```
- 데이터를 확인했을 때 데이터 내부의 값은 모두 숫자로 이루어져 있었다.
    - 정답 데이터는 최대값은 45, 최소값은 0이었고 그 말을 46개의 카테고리로 나누어져 있다고 볼 수 있다.

```python
X_train = sequence.pad_sequences(X_train, maxlen = 100)
X_test = sequence.pad_sequences(X_test, maxlen = 100)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```
- sequence자체의 pedding을 이용해서 최대 길이 100개로 빈 값은 0을 채웠다.
    - 정답 데이터들은 카테고리화 시켰다.

```python
model = Sequential()
model.add(Embedding(1000, 100)) # 1000 -> 100차원으로 축소
model.add(LSTM(100)) # 출력을 100차원으로
model.add(Dense(46, activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, batch_size=20, epochs=200, validation_data=(X_test, y_test), callbacks=[early_stopping_callback])
```
- 모델을 만들고 실행하는 과정이다.
    - 차원을 임베딩을 통해서 1000차원에서 100차원으로 축소했고 출력은 임베딩값인 100차원, 그리고 출력값은 카테고리의 종류 수인 46으로 줬다.
```python
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, y_test)[1]))
71/71 [==============================] - 1s 19ms/step - loss: 1.2641 - accuracy: 0.7173
history = model.fit(X_train, y_train, batch_size=20, epochs=200, validation_split = 0.2, callbacks=[early_stopping_callback])
```
- 참고로 마지막 코드는 validation_split값을 0.2로 주면 X_train, y_train 데이터에 대해 20%를 검증용으로 사용하겠다는 의미다.

```python
# 1D 합성곱
# 자연어 처리에서 1D CNN구조는 LSTM을 이용해서 각 문장은 임베딩을 지나서 임베딩 백터가 된 상태로 LSTM에 입력이 된다. 1D에서도 마찬가지다.
# 문장 토큰화, 패딩, 임베딩층을 거친다. -> 문장의 길이(n) * 임베딩 벡터의 차원(k)
# 1D 합성곱 연산에서 커널의 너비는 문장 행렬에서 임베딩 벡터의 차원과 동일하게 설정한다.
# -> 커널의 높이만으로 해당 커널의 크기라고 간주한다.
# 시퀀스 투 시퀀스
# 입력된 단어들의 모임 (시퀀스)로 부터 다른 도메인의 시퀀스를 출력하는 다양한 분야에서 사용된다
# 대표적으로 기계번역, 챗봇등에서 사용된다.
# 입력 시퀀스와 출력 시퀀스를 질문 답변으로 만들면 챗봇, 입력 문장과 번역 문장으로 만들면 번역기가 된다.
# 음성을 텍스트로 변환하거나 내용 요약등에서도 사용된다.

# 인코더는 모든 단어들을 순차적으로 입력받고, 입력받은 모든 단어 정보들을 압축해서 하나의 벡터로 만든다.
# 디코더는 콘텍스트 벡터를 받아서 번역된 단어를 각각 순차적으로 출력한다.
# 디코더에서 훈련시에는 정답을 받아서 다음 LSTM셀로 이동해 출력하는 방식이다.
# 디코더 테스트시에는 예측값으로 다음 LSTM셀로 이동해 출력하는 방식이다.
# 그 이유는 모델을 학습시키기 때문에 정답으로 다음 값을 예측해야 이상한 값이 나오지 않기 때문이다.
# 이 과정을 교사 강요라고 한다.

```
## RNN으로 주가 예측
```python
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
data = pd.read_csv("tsla.us.txt")
training_size = int(len(data)*0.80)
data_len = len(data)
train, test = data[0:training_size],data[training_size:data_len]
```
- 데이터를 불러와서 그 데이터의 길이 * 0.8을 한 정수를 training_size에 저장했고 데이터의 총 길이를 data_len에 저장했다.
    - 처음부터 training_size까지의 값을 train으로, training_size부터 data_len까지의 값을 test로 지정했다.

```python
print("Training Size --> ", training_size)
print("total length of data --> ", data_len)
print("Train length --> ", len(train))
print("Test length --> ", len(test))

Training Size -->  1486
total length of data -->  1858
Train length -->  1486
Test length -->  372
```

```python
train = train.loc[:, ["Open"]].values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
```
- 시가의 모든 데이터를 train에 저장했고 그 값을 minmax스케일러를 이용해서 스케일링했다.

```python
end_len = len(train_scaled) # 1486
X_train = []
y_train = []
timesteps = 40
```

```python
for i in range(timesteps, end_len):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i,0])
```
- 미리 지정해둔 timesteps를 이용해서 40번째 데이터부터 마지막까지를 범위로 지정해 1~40일까지는 사전데이터로, 41일차는 결과로 지정해줬다.

```python
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
```
- 입력 데이터의 구조를 3차원으로 만들기 위해서 array로 변형시키고 3차원으로 만들어줬다.

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
```

```python
regressor = Sequential()
regressor.add(SimpleRNN(units=50, activation = "tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(SimpleRNN(units=50, activation = "tanh", return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(SimpleRNN(units=50, activation = "tanh", return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(SimpleRNN(units=50, activation = "tanh"))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer= "adam", loss = "mean_squared_error")
epochs = 100
batch_size = 20
regressor.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
```
- 모델을 만들고 학습시키는 과정이다
    - 여기서 다른 부분은 RNN셀의 부분이 한개가 아닌 여러개인 구조라는 것이다.
```python
real_price = test.loc[:, ["Open"]].values
dataset_total = pd.concat((data["Open"], test["Open"]), axis = 0)
```
- test데이터의 정답을 따로 real_price라고 저장해주고 concat을 이용해서 데이터를 합쳐 dataset_total이라고 저장했다.

```python
inputs = dataset_total[len(dataset_total) - len(test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(timesteps, 412):
    X_test.append(inputs[i-timesteps:i,0])
X_test=np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
```
- inputs라는 변수에 전체 데이터수 - 테스트 데이터 수와 40을 뺀값의 인덱스 번호부터 끝까지의 인덱스번호의 해당 데이터를 reshape해 저장했다.
    - for문을 이용해서 각각의 값을 올리고 그 값을 어레이로 변환한 뒤 모델에서 사용하기 위해 3차원으로 만들었다.
```python
pred=regressor.predict(X_test)
pred = scaler.inverse_transform(pred)
```
- 예측값을 pred에 저장하고, inverse_transform을 이용해서 값을 원래 수치로 변환했다.

```python
plt.plot(real_price, color = "red", label = "Real Stock Price")
plt.plot(pred, color = "black", label = "Predict Stock Price")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Tesla Stock Price")
plt.legend()
plt.show()
```
- 그 후 실제값과 비교해 시각화했다.