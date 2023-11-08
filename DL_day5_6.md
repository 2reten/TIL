# CNN

```python
"""
딥러닝의 꽃은 RNN
딥러닝은 주로 예측모델을 만들때 사용이 되어진다
순환 신경망은 이전 셀에서 출력되어진 값이 다음 셀에도 출력이 되어지는 구조다
Vanilla Neural Networks 가장 기본적인 신경망 - one to one 구조
vanilla의 의미는 색소물을 첨가하지 않은 기본적이라는 의미다.
주로 활성화 함수로 "relu"가 사용되고 출력단에서는 "softmax"혹은 "sigmoid"등의 함수로 나뉜다.

RNN은 many to many, many to one, many to many와 many to many가 있다.
입력이 하나가 들어가면 여러개가 출력이 되는구조 - one to many
RNN은 RNN셀에서 바로 출력 계층으로 가는것이 아니라 RNN셀의 다음 셀로 넘어가고 그 셀들을 모두 넘어간 뒤에 출력계층으로 넘어간다.
반대로 입력이 여러개가 있고 출력이 하나인 구조 - many to one
ex) 감성분류 - 영화를 보고 남긴 댓글 (이 영화 참 재미있었다.) 이 문장의 구성이 입력으로 들어가고
그 값을 최종 출력으로 긍정/ 부정으로 나타낸다. 이진분류 문제에서 많이 사용되는 구조
문장을 구성하는 단어들이 입력으로 주어지고 RNN셀에서 연산을 마친 뒤 출력계층에서 출력된다.
여러개가 입력이 되면 출력도 여러개가 출력된다 - many to many
ex)원문은 입력의 시퀸스라면 번역은 출력의 시퀸스라고 할 수 있다.
RNN은 시간에 대한 개념이 확립된 학습법이다.
이 구조는 자연어처리에서 많이 사용된다.
RNN은 사용되어지는 활성화 함수가 다르다. - "tanh"(쌍곡탄젠트)
주가예측에서 사용되는 RNN은 many to one
사전 데이터들이 입력 데이터(many)로 들어가고 출력 데이터(one)를 출력한다.
ex) 3일전 종가가 2일전 종가에 미치는 영향과 2일전의 종가가 하루 전의 종가에 미치는 영향을 보고 오늘의 종가를 예측하는 구조다.
N일전 종가에 대한 데이터는 각각 유가, 정치 관련 뉴스 등 여러 데이터가 들어간다.
RNN셀에서는 3일전의 데이터, 2일전의 데이터, 하루전의 데이터까지 모드 학습을 해 오늘의 종가를 예측한다.
히든계층의 하나 이상의 순환 신경망을 가지고 있느 셀을 RNN이라고 한다.

Image Captioning 구조에서 많이 사용된다
이미지를 보고 해석하는 작업에서 많이 사용되는 구조가 one to many(RNN)구조다.

Sentiment Classification - many to one구조

Machine Translation - many to many
기계 번역
과거에는 한국어와 영어를 서로 번역을 할 때 중간에 일본어를 넣으면 번역이 조금 더 자연스럽게 가능했다.

기본이 되는 네트워크가 RNN이다

Video Classification on Frame level - many to many
프레임 레벨로 비디오를 분류
한편의 비디오는 여러장의 프레임의 집합이라고도 할 수 있다.

동일한 시점에서 w변수(가중치)의 값은 모두 공유된다.
Vocabulary의 크기가 입력 layer의 차원값이다.
hidden layer의 차원은 본인이 직접 정할 수 있다.
출력 layer의 차원 역시 vocabulary의 크기에 따라 적용된다.
ex)입력데이터는 4차원이었으나 입력데이터의 특징들이 3차원의 구조로 표현이 되어진것이다. 

RNN은 충분한 기억력을 가지고 있지 못하기 때문에 엉뚱한 답을 출력할 수 있다.
return_sequences
3D tensor = 배치크기, 타입스탭, 입력차원
RNN셀에 최종 시점의 은닉 상태만을 리턴한다면 batch_size와 output_dim을 크기에 따라 2D tensor를 리턴하지만
만약 모든 은닉 상태값들을 리턴하고자 한다면 똑같이 batch_size, timesteps, output_dim 크기의 3D tensor를 리턴한다.

LSTM
장기의존성 문제를 해결하기 위해서 만들어졌다
cell state는 모든 셀들에 걸쳐 연결되어져 있는 선 - 컨베이어 벨트와 유사
LSTM이 가지고 있는 대표적인 Gate들은 선택적으로 정보를 전달한다.
상황에 맞춰 선택적을 정보를 전달.
LSTM은 망각게이트, 
cell state - 현재 셀을 기준으로 현재 셀의 앞쪽에 대한 상태정보가 들어있다.
망각게이트(forget gate layer) - 전달하는 정보의 양을 조절하는 역할
어떤 문장에서 앞쪽에 있는 전체 문장을 기반으로 다음 단어를 예측하는 언어 모델
어떤 정보를 계속 이어서 전달을 할 것인지와 어떤 정보를 제거를 할 것인지를 조절하는 역할을 한다.
ex) sigmoid의 값이 0이면 전체 제거하라는 의미, 반대로 1이면 모두 저장하라는 의미이다.

입력 게이트 레이어(input gate layer) - 어떤 새로운 정보를 셀 스테이트로 저장을 할 것인가를 결정 할 수 있는 게이트
현재 셀에 입력되는 새로운 정보를 어느정도 저장을 할 것인가를 선정하는 게이트

LSTM은 RNN의 발전된 형태다
LSTM은 시계열 데이터에서 상당히 효과적인 데이터를 도출해낸다.
(자연어 처리에서 많이 사용)
"""
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pip install finance-datareader
import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
```
- CNN을 이용해서 내일의 주식 종가를 알아보기 위한 모듈들이다.
```python
kakao = fdr.DataReader("035720")
kakao["year"] = kakao.index.year
kakao["month"] = kakao.index.month
kakao["day"] = kakao.index.day
```
- 카카오의 주식 데이터를 가져오고 년, 월, 일을 값으로 새로운 열을 만들었다.

```python
plt.figure(figsize = (16,9))
sns.lineplot(y=kakao["Close"], x = kakao.index)
plt.xlabel("time")
plt.ylabel("price")
```
- 종가를 시간의 흐름에 따라 값을 선 그래프로 만든것이다.

```python
ts = [["2000", "2010"],
      ["2010", "2015"],
      ["2015", "2020"],
      ["2020", "2025"]]
fig, axes = plt.subplots(2,2)
fig.set_size_inches(16,9)
for i in range(4):
    ax = axes[i//2, i%2]
    df = kakao.loc[(kakao.index > ts[i][0]) & (kakao.index <ts[i][1])]
    sns.lineplot(y=df['Close'], x=df.index, ax=ax)
    ax.set_title(f'{ts[i][0]}~{ts[i][1]}')
    ax.set_xlabel('time')
    ax.set_ylabel('price')
plt.tight_layout()
```
- for문을 이용해서 시간대별 그래프를 한번에 시각화했다.
```python
scaler = MinMaxScaler()
cols = ['Open', 'High', 'Low', 'Close', 'Volume']
scaled = scaler.fit_transform(kakao[cols])
df = pd.DataFrame(scaled, columns=cols)
```
- MinMaxScaler를 사용하기 위해서 필요한 값들만 가져와 그 값을 scaled로 저장한 뒤 데이터프레임으로 만들었다.
```python
xtrain, xtest, ytrain, ytest = train_test_split(df.drop("Close",1), df["Close"], test_size = 0.2, random_state = 0, shuffle = False)
```
- train과 test로 데이터를 나눴다.

# RNN

```python
def make_dataset(data, label, window_size=20):
    # print(data.shape) # (4736, 4)
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size): # 4736 - 20 = 4716 => i = 0~4715
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)
xtrain, ytrain = make_dataset(xtrain, ytrain, 20)
```
- 한 입력값에 들어갈 데이터의 사이즈를 20으로 지정하고 data에 훈련데이터를 label에 정답 데이터를 넣었다.
    - feature_list에는 4715, 4 각 값에는 20개의 데이터가 저장되어있다.
    - lalbel_list에는 4715, 의 크기로 저장되어 있고 데이터는 20개씩 묶여있다.
```python
ytrain = ytrain.reshape(ytrain.shape[0], 1)
xtest, ytest = make_dataset(xtest, ytest, 20)
ytest = ytest.reshape(ytest.shape[0], 1)
```
- train과 test모두 같은 유형으로 만들어주고 뒤에 1값을 추가해서 만들었다.

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
```
```python
model = Sequential()
model.add(LSTM(16, 
               input_shape = (xtrain.shape[1], xtrain.shape[2]),
               activation = "relu",
               return_sequences = False
              ))model.add(LSTM(16, 
               input_shape = (xtrain.shape[1], xtrain.shape[2]),
               activation = "relu",
               return_sequences = False
              ))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('tmp_checkpoint.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
xtrain, x_valid, ytrain, y_valid = train_test_split(xtrain, ytrain, test_size=0.2)
history = model.fit(xtrain, ytrain, 
                    epochs=200, 
                    batch_size=16,
                    validation_data=(x_valid, y_valid), 
                    callbacks=[early_stop, checkpoint])
```
- 모델을 만들고 훈련을 시키는 과정이다.
    - 먼저 출력값은 16으로 지정하고 활성함수는 "relu" 그리고 값은 1개만 출력하면 되기에 return_sequences는 false로 지정했다.
    - Dense열에 활성함수를 주지 않은것은 default 값인 linear함수를 사용할 것이기 때문이다.
    - 아래 compile과 fit은 지금까지와 같은 방식이다.
```python
model.load_weights("tmp_checkpoint.h5")
pred = model.predict(xtest)
plt.figure(figsize=(12, 9))
plt.plot(ytest, label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()
```
- 모델의 예측값을 출력하고 그 값과 실제 값의 차이를 확인했다.

## SimpleRNN
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2,10)))
model.summary()
```
- simpleRNN 코드다.

```python
import numpy as np

timesteps = 10
input_dim = 4
hidden_units = 8

inputs = np.random.random((timesteps, input_dim))

hidden_state_t = np.zeros((hidden_units,)) 

print('초기 은닉 상태 :',hidden_state_t)
```
- model.add(SimpleRNN(3, input_length=2, input_dim=10))와 동일하다.
    - inputs값은 입력에 해당되는 2D 텐서다.
    - hidden_units는 초기 은닉 상태로 0(벡터)으로 초기화했다.

```python
Wx = np.random.random((hidden_units, input_dim)) 
Wh = np.random.random((hidden_units, hidden_units)) 
b = np.random.random((hidden_units,))
print('가중치 Wx의 크기(shape) :',np.shape(Wx))
print('가중치 Wh의 크기(shape) :',np.shape(Wh))
print('편향의 크기(shape) :',np.shape(b))
```
- Wx는 (8, 4)크기의 2D 텐서 생성하고 입력에 대한 가중치를 의미한다.
- Wh는 (8, 8)크기의 2D 텐서 생성하고 은닉 상태에 대한 가중치를 의미한다.
- b는 (8,)크기의 1D 텐서 생성하고 이 값은 편향(bias)을 의미한다.

```python
total_hidden_states = []

for input_t in inputs:

 
  output_t = np.tanh(np.dot(Wx,input_t) + np.dot(Wh,hidden_state_t) + b)

 
  total_hidden_states.append(list(output_t))
  hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis = 0) 

# (timesteps, output_dim)
print('모든 시점의 은닉 상태 :')
print(total_hidden_states)
```
- 각 시점 별 입력값을 출력하는 코드다.
    - Wx * Xt + Wh * Ht-1 + b(bias)은 output_t의 공식이다.
    - 각 시점 t별 메모리 셀의 출력의 크기는 (timestep t, output_dim)이고, 각 시점의 은닉 상태의 값을 계속해서 누적한다.
    - 출력 시 값을 깔끔하게 해주는 용도로 np.stack(total_hidden_states, axis = 0)을 적용해 다시 저장했다.
```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.utils import to_categorical
```
- RNN은 모든 타임스탭이 동일해야 함 부족한 값은 0으로 채운다.
    - to_categorical은 원핫인코딩으로 만들어준다.
```python
text = """경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고와야 오는 말이 곱다\n"""
tokenizer = Tokenizer() # 문장을 토큰 단위로 나눔
tokenizer.fit_on_texts([text]) # 문자열은 리스트구조로 변경해 
vocab_size = len(tokenizer.word_index) + 1
print('단어 집합의 크기 : %d' % vocab_size)
```
- 문장을 text에 정의하고 토큰화시켰다.
    - 그리고 vocab_size는 토큰의 길이 + 1값을 줬다
        - 인덱스 0번없이 1번부터 시작하기 때문이다.
```python
sequences = list()
for line in text.split('\n'): # 줄바꿈 문자를 기준으로 문장 토큰화
    encoded = tokenizer.texts_to_sequences([line])[0] # word_index를 기준으로 인코딩 된 값을 출력한다.
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print('학습에 사용할 샘플의 개수: %d' % len(sequences))
```
- 이 코드는 줄바꿈 문자를 기준으로 문장 토큰화를 하고 word_index를 기준으로 인코딩 된 값을 출력한다.

```python
max_len = max(len(l) for l in sequences) # 모든 샘플에서 길이가 가장 긴 샘플의 길이 출력
print('샘플의 최대 길이 : {}'.format(max_len))
```
- 샘플의 최대 길이 즉, 문장이 가장 긴 문장의 토큰 길이를 출력하는 것이다.

```python
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y, num_classes = vocab_size)
```
- 여기서 padding은 pre와 post가 있는데 pre를 지정하면 최대 길이 수를 기준으로 비는 공백의 칸은 0을 앞으로 채우고, post는 뒤로 채운다.
    - array로 형식으로 변형하고 정답열과 데이터열을 구분해서 각각 X,y로 저장한 뒤 y는 원 핫 인코딩을 해줬다.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
```

```python
embedding_dim = 10 # 임베딩 차원 : 10
# 임베딩? 단어를 벡터 공간에 표현하는 것
# 임베팅 벡터 공간 : 11차원(단어의 종류 개수) + 1 = 12차원
hidden_units = 32

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))

model.add(SimpleRNN(hidden_units))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)
```
- 임베딩 차원 : 10
    - 임베딩? 단어를 벡터 공간에 표현하는 것
    - 임베팅 벡터 공간 : 11차원(단어의 종류 개수) + 1 = 12차원
- Embedding은 12차원 데이터를 10차원 공간 데이터로 표현했다.
- 12 -> 10차원으로 변경하면 원 핫 인코딩 되어져 있던 값들이 실수형태로 변환된다. [00100..0] => [1.3 -1.1 -0.3 ... 3]
- 임베딩을 하는 이유는 고차원의 단어 벡터로 구성되어져 있는 단어 벡터를 저차원으로 줄여줘 학습속도를 개선하는 용도다.

```python
def sentence_generation(model, tokenizer, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word
    sentence = ''

    # n번 반복
    for _ in range(n):
        # 현재 단어에 대한 정수 인코딩과 패딩d
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=5, padding='pre')
        # 입력한 X(현재 단어)에 대해서 Y를 예측하고 Y(예측한 단어)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for word, index in tokenizer.word_index.items(): 
            if index == result:
                break

        current_word = current_word + ' '  + word

        sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence
```
- for문을 이용해서 모델 값을 예측하고 예측한 단어와 인덱스가 동일한 단어가 있다면 break, 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경, 그리고 예측 단어를 문장에 저장한 뒤 값을 출력시켰다.

```python
print(sentence_generation(model, tokenizer, '경마장에', 4))
경마장에 있는 말이 뛰고 있다
```
```python
import pandas as pd
import numpy as np
from string import punctuation

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
```
```python
df = pd.read_csv('ArticlesApril2018.csv')
print('열의 개수: ',len(df.columns))
print(df.columns)
print(df['headline'].isnull().values.any())
```
- 데이터를 불러오고 열의 개수와 결측값의 유무를 파악했다.

```python
headline = []
headline.extend(list(df.headline.values))
print('총 샘플의 개수 : {}'.format(len(headline)))
headline = [word for word in headline if word != "Unknown"]
print('노이즈값 제거 후 샘플의 개수 : {}'.format(len(headline)))
```
- extend를 이용해서 벨류를 값으로 추가하고, 총 샘플의 개수와 노이즈 제거후의 샘플의 수를 각각 출력 시켰다.
    - 총 샘플의 개수 : 1324 -> 노이즈값 제거 후 샘플의 개수 : 1214

```python
def repreprocessing(raw_sentence):
    preproceseed_sentence = raw_sentence.encode("utf8").decode("ascii",'ignore')
    return ''.join(word for word in preproceseed_sentence if word not in punctuation).lower()

preprocessed_headline = [repreprocessing(x) for x in headline]
preprocessed_headline[:5]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_headline)
vocab_size = len(tokenizer.word_index) + 1
print('단어 집합의 크기 : %d' % vocab_size)
```
- 구두점을 제거하며 동시에 데이터를 소문자로 변환시켰다.
    - 단어 집합의 크기를 파악했다.
        - 단어 집합의 크기 : 3494

```python
sequences = list()

for sentence in preprocessed_headline:

    encoded = tokenizer.texts_to_sequences([sentence])[0] 
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)
sequences[:11]
```
- 각 샘플에 대한 정수를 인코딩했다.

```python
index_to_word = {}
for key, value in tokenizer.word_index.items():
    index_to_word[value] = key
```
- 인덱스를 단어로 바꾸기 위해 index_to_word를 생성했다.

```python
max_len = max(len(l) for l in sequences)
print('샘플의 최대 길이 : {}'.format(max_len))
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
y = to_categorical(y, num_classes = vocab_size)
```
- 최대길이를 파악한 후 그 길이를 기준으로 padding을 pre값을 줘서 빈칸의 앞에는 0을 채우고 각각 정보 데이터와 정답 데이터를 나눠서 X,y로 나눴다.
    - y는 원핫인코딩으로 만들어줬다.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
embedding_dim = 10
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim)) # 3494 -> 10
model.add(LSTM(hidden_units)) # LSTM 셀 출력 : 128차원
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)
```
- 임베딩으로 차원을 축소했다.
    - 출력 차원으로는 128차원을 지정했고, 이 모델은 분류를 하는 것이기 때문에 softmax를 활성화 함수로 사용했다.
```python
def sentence_generation(model, tokenizer, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word
    sentence = ''

    # n번 반복
    for _ in range(n):
        encoded = tokenizer.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=max_len-1, padding='pre')

        # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        result = model.predict(encoded, verbose=0)
        result = np.argmax(result, axis=1)

        for word, index in tokenizer.word_index.items(): 
            # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
            if index == result:
                break

        # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        current_word = current_word + ' '  + word

        # 예측 단어를 문장에 저장
        sentence = sentence + ' ' + word

    sentence = init_word + sentence
    return sentence
print(sentence_generation(model, tokenizer, 'i', 10))
print(sentence_generation(model, tokenizer, 'how', 10))
```
- 출력값으로는 i cant jump ship from facebook yet syria them war choose, how do you feel about being told to smile parents mean로 각각 출력됐다.