```python
# Seq2seq는 글자, 이미지등의 아이템의 sequence를 받아서 다른 언어로 번역된 sequence를 출력한다.
# encoder와 ddecoder로 이루어져있고
# encoder는 입력된 정보에서 정보를 추출해 추출한 정보를 벡터로 변환한다.
# 그 변환한 context벡터를 decoder로 보낸다.
# decoder는 그 아이템을 하나씩 출력한다.
# 기계번역에서도 구조로 번역이 이루어진다.
# context 벡터는 실수형태의 값이 저장되어져 있는 벡터라고 할 수 있다.
# context 벡터의 크기를 설정하고자 할 때, 일반적으로는 encoder안에 hidden units의 값과 동일하게 준다.
# Seq2seq
# 하나의 RNN마다 두개의 입력값을 받는다.
# 워드 임베딩을 통해 단어들을 벡터 공간에 표현할 수 있다.

# Transformer
# 트렌스포머를 블랙박스라고 생각을 해보면, 기계번역시에 모델은 어떤 하나의 언어로 된 문장을 입력 받아서 다른 언어로 된 결과를 도출한다.
# 그렇다면 중간에 있는 트랜스포머는 결국 인코딩과 디코딩을 필연적으로 하게 된다.
# 다만, 인코더와 디코더의 개수는 한개가 아닌 여러개다. 인코더와 디코더의 개수는 같아야한다.
# 모든 인코더 들은 동일한 구조를 가진다. 허나, 구조가 같다는 말이 가중치를 공유한다는 말은 아니다. 각각 다른 가중치를 가진다.
# 하나의 인코더 안에는 두개의 서브 레이어가 있다. (self-attention, feed forward neural network)
# self attention에서는 다른 단어들과의 관계를 살펴보는 계층이다.
# 가장 하단의 인코더에서는 워드 임베딩이 수행된다.
# 인코더로부터 임베딩 된 것은 다른 인코더로 출력이 된다.
# Feed Forward layer에서는 다양한 paths들로부터 입력되는 단어들에 대해서 병렬구조로 처리를 하는 것이 가능하다.
# Self-attention
# 인코더의 입력된 벡터로부터 3개의 벡터가 만들어진다.
# 그 벡터들을 Query vector, Key vector, Value vector라고 한다.
# 이 3개의 벡터들이 나중에는 학습 가능한 행렬과의 곱셈을 하며 만들어진다.
# 기존에 존재하는 임베딩 벡터보다 차원은 더 작다.
# multiheaded attention의 계산을 더욱 일정하게 만들기 위해서 설계된 것이다.
# 임베딩값과 가중치 행렬을 곱하면 각각 query, key, value 행렬이 만들어진다.
```

```python
import os
import shutil
import zipfile
import requests

import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
```
```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def download_zip(url, output_path):
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): # chunk size는 8192건씩 가져와 읽으라는 의미
                f.write(chunk)
        print(f"ZIP file downloaded to {output_path}")
    else:
        print(f"Failed to download. HTTP Response Code: {response.status_code}")

url = "http://www.manythings.org/anki/fra-eng.zip"
output_path = "fra-eng.zip"
download_zip(url, output_path)

path = os.getcwd() # 현재 디렉토리 경로
zipfilename = os.path.join(path, output_path) 

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)
```
- 함수를 만들어 주소를 받아 파일을 다운한다.

```python
lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
print('전체 샘플의 개수 :',len(lines))
```
- 전체 샘플의 개수 : 227815
```python
lines = lines[0:60000] 
lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')
```
- 6만개의 데이터만을 저장하고 tar열에 시작부분에는 \t를 끝부분에는 \n을 추가했다.

```python
src_vocab = set()
for line in lines.src:
    for char in line:
        src_vocab.add(char)
tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)
```
- 1줄씩 읽어 그 문장들을 한 문자씩 읽어 unique값만 추가했다.

```python
src_vocab_size = len(src_vocab)+1
tar_vocab_size = len(tar_vocab)+1
print('source 문장의 char 집합 :',src_vocab_size)
print('target 문장의 char 집합 :',tar_vocab_size)
```
- source 문장의 char 집합 : 80
target 문장의 char 집합 : 104

```python
src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
```
- 각 리스트를 아스키코드값 순으로 정렬시켰다.

```python
src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
```
- 딕셔너리 구조로 각 문자에 id를 만들어줬다.

```python
encoder_input = []
for line in lines.src:
    encoded_line = []
    for c in line:
        encoded_line.append(src_to_index[c])
    encoder_input.append(encoded_line)
decoder_input = []
for line in lines.tar:
    encoded_line = []
    for char in line:
        encoded_line.append(tar_to_index[char])
    decoder_input.append(encoded_line)
print('target 문장의 정수 인코딩 :',decoder_input[:5])
```
- encoded_line에는 src의 인덱스 번호를, encoder_input에는 encoded_line을 넣었다.
- decoder도 마찬가지로 처리해줬다.
    - target 문장의 정수 인코딩 : [[1, 3, 48, 52, 3, 4, 3, 2], [1, 3, 39, 52, 69, 54, 59, 56, 14, 3, 2], [1, 3, 31, 65, 3, 69, 66, 72, 71, 56, 3, 4, 3, 2], [1, 3, 28, 66, 72, 58, 56, 3, 4, 3, 2], [1, 3, 45, 52, 63, 72, 71, 3, 4, 3, 2]]

```python
decoder_target = []
for line in lines.tar:
    timestep = 0
    encoded_line = []
    for char in line:
        if timestep > 0:
            encoded_line.append(tar_to_index[char])
        timestep = timestep + 1
    decoder_target.append(encoded_line)
print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])
```
- target 문장 레이블의 정수 인코딩 : [[3, 48, 52, 3, 4, 3, 2], [3, 39, 52, 69, 54, 59, 56, 14, 3, 2], [3, 31, 65, 3, 69, 66, 72, 71, 56, 3, 4, 3, 2], [3, 28, 66, 72, 58, 56, 3, 4, 3, 2], [3, 45, 52, 63, 72, 71, 3, 4, 3, 2]]

```python
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
print('source 문장의 최대 길이 :',max_src_len)
print('target 문장의 최대 길이 :',max_tar_len)
```
- padding은 각각 source는 source끼리 동일하게 target은 target끼리만 동일하게 해주면 된다.

```python
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')
```
- 각각 패딩을 하는 과정이다.

```python
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np
```
- 카테고리화 했다.

```python
encoder_inputs = Input(shape=(None, src_vocab_size)) 
```
- src_vocab_size 입력되는 문자의 종류 => 문자 벡터의 크기
- None은 문자열의 길이를 의미한다. 
- None값을 주면 Input이 알아서 판단을 한다.

```python
encoder_lstm = LSTM(units = 256, return_state = True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]
```
- state_h(은닉상태), state_c(셀상태)이다.

# Seq2Seq 디코더 설계
```python
decoder_inputs = Input(shape=(None, tar_vocab_size)) 
decoder_lstm = LSTM(units = 256, return_state = True, return_sequences = True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state = encoder_states)
decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs],decoder_outputs)
model.compile(optimizer = "rmsprop", loss = "categorical_crossentropy")
model.fit(x = [encoder_input, decoder_input], y = decoder_target, batch_size = 64,
          epochs = 40, validation_split = 0.2)
```
- 평소와 다른 방식으로 모델을 설계했다.
    - _, _는 버리는 값이라는 의미다.
- 테스트 과정
- 1. 번역 대상 입력 문장이 인코더에 들어감 -> 은닉/셀 상태 -> 디코더로 전달
- 2. 디코더의 입력 시그널이 전달(sos, \t)
- 3. 디코더는 다음 문자 예측

## 테스트 과정에서의 디코더 정의

```python
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)
```
- 이전 시점의 상태들을 저장하는 텐서들을 각각 decoder~들로 저장했다.
    -  문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용했고, 뒤의 함수 decode_sequence()에 동작을 구현 예정이다.
    - 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태를 버리지 않는다.
```python
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())
```
- 시간 문제로 설명을 듣지 못한 코드다.