```python
"""
LSTM 분류기
Yolo - 사진 객체 추출

다음주
- 어텐션, 트랜스포머, BERT, GPT
- 파인튜닝
-> 이미지(cnn), 텍스트(bert, gpt)
- yolo 개인 데이터 객체 추출
- tableau 기초, 대시보드
다음주 이후
html, javascript, flask, db

프로젝트 예정

파이널 프로젝트 기간
- 오전 수업, 오후 프로젝트
- GAN, 강화학습, 메타러닝 등
"""
```
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```
```python
data = pd.read_csv('spam.csv', encoding='latin1')
```
- 글자가 깨져서 encoding='latin1'값을 줬다.

```python
del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
```
- 결측값밖에 없는 2,3,4열은 지웠다.
```python
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
print('v2열의 유니크한 값 :',data['v2'].nunique())
```
- replace([변경전],[변경후])
- 유니크값을 확인하는 코드다.

```python
data.drop_duplicates(subset=['v2'], inplace=True) 
```
- 중복 제거할때 drop_duplicates쓰면 편하다.
```python
X_data = data['v2']
y_data = data['v1']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0, stratify=y_data)

```
- 학습 데이터를 구분하기 위해서 정답열과 나눠서 각각 X_data, y_data로 저장했다.
- stratify = spam과 ham의 비율을 유지하면서나눠줌 => 층화추출

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
```
- 피팅 = 코퍼스에 있는 텍스트 데이터로부터 토크나이저를 생성했다.
```python
X_train_encoded=tokenizer.texts_to_sequences(X_train)
```
- X_train에 저장되어있는 데이터를 시퀀스로, 인덱스번호로 보여줬다.
```python
word_to_index = tokenizer.word_index
len(word_to_index)
```
- X_train에 저장되어있는 데이터를 시퀀스로, 인덱스번호로 보여줬다.
```python
tokenizer.word_counts.items() 
```
- 모델의 크기도 줄이고 빈도수 낮은 단어는 빼기도한다
```python
total_cnt = len(word_to_index)
threshold = 2
rare_cnt = 0 
total_freq = 0 
rare_freq = 0 
```
- 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트(rare_cnt)
- 훈련 데이터의 전체 단어 빈도수 총 합(total_freq)
- 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합(rare_freq)
```python
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value
    if(value < threshold): 
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
```
-  키에는 단어가 벨류에는 단어수가 차례로 들어간다.
```python
rare_cnt/total_cnt 
```
- 전체 단어중에서 등장 빈도수가 1인 단어의 비율이다.
```python
(rare_freq/total_freq)*100
```
- 전체 등장 빈도수에서 등장 빈도수가 1인 단어의 등장 비율이다.
- 등장 빈도가 매우낮은 단어는 자연어처리에서 제거해 버리는 경우도 있다.
```python
tokenizer_over2 = Tokenizer(num_words=total_cnt-rare_cnt+1)
tokenizer_over2.fit_on_texts(X_train)
```
- total_cnt-rare_cnt + 1은 빈도를 의미한다.

```python
# 잠깐 간단한 예제로 num words기능 확인
# sentences = [
#     'I love my dog',
#     'I, love my cat',
#     'You love my dog!'
# ]

# tokenizer3 = Tokenizer(num_words=4)
# tokenizer3.fit_on_texts(sentences)
# word_index = tokenizer3.word_index
# word_index #빈도수 높은순으로 되어있는데 넘워즈 4로하면 1~3번까지만 사용하겠다
# seq = tokenizer3.texts_to_sequences(sentences)
# print(word_index)  # {'love': 1, 'my': 2, 'i': 3, 'dog': 4, 'cat': 5, 'you': 6}
# print(seq) #넘워즈 4라고해서 3개만나옴
# 여기까지 num words기능 확인 위한 간단한예제
# 넘워즈 설정해주면 빈도수 높은거 순대로 설정해준만큼 나옴
```
```python
print('메일의 최대 길이 : %d' % max(len(sample) for sample in X_train_encoded))
print('메일의 평균 길이 : %f' % (sum(map(len, X_train_encoded))/len(X_train_encoded)))
```
- 메일의 최대 길이 : 189
- 메일의 평균 길이 : 15.754534
```python
plt.hist([len(sample) for sample in X_data], bins=50) 
```
- 구간을 50개로 나누어 hist로 표현했다.
```python
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential
X_train_padded = pad_sequences(X_train_encoded, maxlen =189)

embedding_dim = 32
hidden_units = 32
```
```python
vocab_size=len(word_to_index)+1 
vocab_size 
```
- 패딩 토큰 0번이 사용되므로 1을 더했다
- 단어가 0번이없고 1번부터 7821까지인데 패딩 토큰0번이 사용되므로1을 더헀다.
```python
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim)) #7822 -> 32차원
model.add(SimpleRNN(hidden_units))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train_padded, y_train, epochs=4, batch_size=64, validation_split=0.2)
```
- optimizer= 아담이아닌 'rmsprop'를 줬다.
- 이진분류이므로 loss='binary_crossentropy'써야한다.
- x_train_padded데이터에 20퍼정도 나눠서 validation하는거로 사용했다.
```python
# 팀원별 메일 제목 / 분류결과 데이터셋 구성
# - 스팸 메일 / 햄 메일 분류기
# - 도착 메일 -> 누구의 메일일까? 자동 분류
# 메일 예민한거 빼고 다 오픈해서 메일제목 스크래핑 
```
```python
X_test_encoded=tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_encoded, maxlen =189)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test_padded, y_test)[1]))
33/33 [==============================] - 0s 6ms/step - loss: 0.1756 - acc: 0.9275

 테스트 정확도: 0.9275
```
```python
model.predict(X_test_encoded)
33/33 [==============================] - 0s 6ms/step
array([[0.01302956],
       [0.01326678],
       [0.98749626],
       ...,
       [0.04057339],
       [0.9914956 ],
       [0.01572474]], dtype=float32)
```
```python
# "고객님 이번에 세일을 합니다. 방문해주세요"
# -> 토큰화 -> 수치 변환 -> 패딩 -> 모델 입력 ->

# 웹앱형식 서비스
# 모델에 대한 정성적 평가
# XAI = eXplainable AI

# 스팸/햄 메일 분류기 자동화
# - 데이터 수집 -> 전처리 -> 분석 -> 시각화 -> 모델링 -> 배포 -> 성능개선
# - 기존 모델에 새롭게 수집된 데이터를 추가하여 학습 # 파인튜닝
# -0.01 -> 0, 0.49999
```
```python
from konlpy.tag import Okt
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```
```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")
total_data = pd.read_table('ratings_total.txt', names=['ratings', 'reviews'])
```
- urllib을 이용해서 페이지에서 바로 데이터를 저장한 뒤, 그 데이터를 불러왔다.
```python
total_data=total_data.head(10000)
total_data['label']=np.select([total_data.ratings>3], [1], default=0)
```
- 데이터를 상위 10000개만 선별했고, select를 이용해서 ratings가 3보다 큰 값들을 선정해  label이라는 열을 추가해 넣었다.
    - ratings가 3 이상이면 1을 가지고 아니라면 0을 가진다.
```python
total_data.drop_duplicates(subset=['reviews'], inplace=True)
train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42)
```
- drop_duplicates를 이용해서 중복값을 지워 저장했고 그 후 트레인과 테스트로 데이터를 나눠 저장했다.
```python
train_data['reviews']=train_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
```
- 모든 한글과 공백을 빼고 모두 제거해 저장했다.
```python
train_data['reviews'].replace("", np.nan, inplace=True)
```
- replace를 이용해서 결측값으로 대체해 저장했다.
```python
test_data.drop_duplicates(subset=['reviews'], inplace=True)
test_data['reviews']=test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
test_data['reviews'].replace("", np.nan, inplace=True)
test_data=test_data.dropna(how='any')
```
- 코드를 정리한 것이다.
```python
okt=Okt()
okt.morphs('배송도 빠르네요 가격대비 좋은것 같아요 첨에는 힘들어하나 조금 지나니 잘 하네요')
okt.pos('배송도 빠르네요 가격대비 좋은것 같아요 첨에는 힘들어하나 조금 지나니 잘 하네요')
```
- morphs는 형태소 단위로 분류하는 코드고, pos는 형태소와 더불어 품사까지 출력을 하는 속성이다.
```python
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게']
train_data['tokenized']=train_data['reviews'].apply(okt.morphs)
train_data['tokenized']=train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
```
- 불용어를 지정하고 형태소 단위로 구분을 해 tokenized라는 열에 저장했다.
    - 그 후 불용어를 제거했다.
```python
from collections import Counter 
```
- 각 단어의 빈도수를 출력해주는 모듈이다.
```python
list = ['Hello', 'HI', 'How', 'When', 'Where', 'Hello']
Counter(list)
```
- Counter({'Hello': 2, 'HI': 1, 'How': 1, 'When': 1, 'Where': 1})
```python
negative_words = np.hstack(train_data[train_data.label==0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data.label==1]['tokenized'].values)
```
- 부정 단어들을 hstack을 이용해서 label값이 0인 토큰들의 데이터들을 저장하고, 긍정 단어들은 label값이 1인 토큰들의 데이터들을 저장했다.
```python
Counter(negative_words)
negative_word_count = Counter(negative_words)
Counter(positive_words)
positive_words_count = Counter(positive_words)
```
- 각각 빈도수를 측정하고 그 값을 저장했다.
```python
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5)) #섭플랏츠 하줄에 두칸
text_len = train_data[train_data['label']==1]['tokenized'].map(lambda x: len(x))
ax1.hist(text_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('긍정 리뷰의 평균 길이 :', np.mean(text_len))

text_len = train_data[train_data['label']==0]['tokenized'].map(lambda x: len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Negative Reviews')
fig.suptitle('Words in texts')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
print('부정 리뷰의 평균 길이 :', np.mean(text_len))
plt.show()
```
- 긍정 리뷰와 부정 리뷰를 비교하기 위해 hist로 표현했다.
```python
test_data['tokenized'] = test_data['reviews'].apply(okt.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
```
- 같은 과정이다. 형태소로 분리해 토큰화 후 불용어를 제거하는 과정이다.
```python
threshold = 2
total_cnt = len(tokenizer.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
```
- 단어 집합(vocabulary)의 크기 : 16865
등장 빈도가 1번 이하인 희귀 단어의 수: 10083
단어 집합에서 희귀 단어의 비율: 59.78654017195375
전체 등장 빈도에서 희귀 단어 등장 빈도 비율: 10.609663706384948

```python
vocab_size = total_cnt - rare_cnt + 2
```
-  0 패딩 토큰, OOV(out of vocabulary,사전에 없는 단어) 토큰을 추가했다.

```python
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
```
- 토큰화 한 것을 sequences로 만들어 값을 저장했다.

```python
print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
```
- 리뷰의 최대 길이 : 51
리뷰의 평균 길이 : 12.671466666666667

```python
X_train = pad_sequences(X_train, maxlen = 51) 
X_test = pad_sequences(X_test, maxlen = 51)
```
```python
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

```python
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
```
- 모델을 만들고 임베딩을 이용해서 차원을 100차원으로 축소했다.

```python
es = EarlyStopping(monitor='val_loss', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
loaded_model = load_model('best_model.h5')
loaded_model.evaluate(X_test, y_test)[1]
```
- earlystopping 값을 설정하고, modelcheckpoint는 Keras 모델을 학습하는 동안 일정한 간격으로 모델의 가중치를 저장하고, 최상의 성능을 보인 모델을 선택하는 기능이다.
    - 79/79 [==============================] - 1s 13ms/step - loss: 0.4012 - acc: 0.8572
0.857200026512146