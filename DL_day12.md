```python
# 센텐스버트는 버트의 문장 임베딩의 성능을 우수하게 개선시킨 모델이다.
# 버트의 문장 임베딩을 이용해서 파인튜닝시킨다.

# 트렌스포머스 라이버르러리에는 각종 태스크에 맞게 벌트모델위에 출려긍을 추가한 모델 클래스 구현체를 제공한다.
# 다대일 유형, 다대다 유형, 질의 응답 유형이 있다.
# 다대일은 감성분석과 같은 유형에 사용되고 다대다는 번역기, 질의 응답 유형은 챗봇과 같은 것들이 해당된다.
# 다대일은 주로 텍스트 분류에 사용된다. (TFBertForSequenceClassificatioin)
# 다대다는 주로 개체명 인식이 해당 모델을 사용하는 대표적인 예시다.

# BERT : 구글에서 104개 이상 국가의 언어로 만든 모델이다.
# 사전 학습된 한국어 BERT모델
# KorBRT : ETRI(20gb 이상의 데이터, 3만개 이상의 단어)
# KoBERT, KoBART : SKT
# ...
# KoGPT : SKT
# KoreALBERT : 나무위키와 뉴스등에서 데이터를 수집해 40gb정도의 데이터로 이루어져있다.
# HyperCLOVA : 네이버, 초대규모 모델
# KLUE -BERT
# KoGPT : 카카오
# ...

# 정수 인코딩, 세그먼트 인코딩, 어텐션 마스크가 필요하다.
# 정수 인코딩 = [2, 1537, 2534, 2069, 6572, 2259, 3771, 18, 3690, 4530, 2585, 2073, 3771, 3]
# 세그먼트 인코딩은 [0] * max_seq_len
# 어텐션 마스크 : 토큰 위치는 1, 패딩 위치는 0이다. [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...0]
```
## 센텐스버트
```python
import tensorflow as tf
import os
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)
```
- colab에서 TPU를 연결하는 코드다.

```python
def create_model():
  return tf.keras.Sequential(
      [tf.keras.layers.Conv2D(256, 3, activation='relu', input_shape=(28, 28, 1)),
       tf.keras.layers.Conv2D(256, 3, activation='relu'),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(256, activation='relu'),
       tf.keras.layers.Dense(128, activation='relu'),
       tf.keras.layers.Dense(10)])
with strategy.scope():
  model = create_model()
  model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
```
- TPU를 사용하며 딥러닝 모델을 만들때는 add.을 이용한것이 아닌 리스트에 모두 담아서 Sequential 클래스에 담아 리턴하는 함수를 만드는 것이 좋다.
    - 모델을 만들고 컴파일 할 때 with strategy.scope(): 구문을 먼저 작성해는 것이 좋다.
    - 이 작성안이 일반적인 약속이다.

## BERT, TPU, 네이버 영화 댓글 분류기

```python
pip install transformers
import transformers
import pandas as pd
import numpy as np
import urllib.request
import os
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
```

```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
```
- urllib을 이용해 사이트에서 데이터를 가져와 각각 train_data, test_data로 저장했다.

```python
train_data = train_data.dropna(how = 'any')
train_data = train_data.reset_index(drop=True)
print(train_data.isnull().values.any()) 
```
-  Null 값이 존재하는 행을 제거했다.
    - 그 후 인덱스 번호를 초기화한 뒤, Null 값이 존재하는지 확인했다.
```python
test_data = test_data.dropna(how = 'any')
test_data = test_data.reset_index(drop=True)
print(test_data.isnull().values.any())
```
- test데이터도 같은 작업을 해줬다.

```python
train_data.drop_duplicates(subset=['document'], inplace=True)
```
- 중복행을 제거하는 코드다.
```python
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
```
- 한국어 bert모델을 불러왔다.

```python
print(tokenizer.tokenize("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))
print(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역"))
```
- ```python
    ['보', '##는', '##내', '##내', '그대로', '들어맞', '##는', '예측', '카리스마', '없', '##는', '악역']
    [2, 1160, 2259, 2369, 2369, 4311, 20657, 2259, 5501, 13132, 1415, 2259, 23713, 3]
```
값이 출력됐다.

```python
print(tokenizer.decode(tokenizer.encode("보는내내 그대로 들어맞는 예측 카리스마 없는 악역")))
```
- 디코딩을 다시 하면 문장의 시작과 끝에 각각 [CLS], [SEP]가 붙는것을 알 수 있다.

```python
max_seq_len = 128
encoded_result = tokenizer.encode(" 전율을         일으키는         영화 . 다시         보고싶은   영화 " , max_length=max_seq_len, pad_to_max_length=True)
```
- max_seq_len를 지정해주고 문장을 인코딩했다.
    - 문장을 토큰화하고 max_seq_len값만큼 남은 부분은 모두 0으로 패딩했다.

```python
valid_num = len(tokenizer.encode("전율을 일으키는 영화. 다시 보고싶은 영화"))
print(valid_num * [1] + (max_seq_len - valid_num) * [0])
```
- valid_num은 문장을 토큰화 한 것의 길이로 지정했고, valid_num길이만큼 [1]을 표시하고, 뒷 부분은 max_seq_len - valid_num만큼 [0]으로 표시했다.
    - 패딩의 과정이고 이것은 어텐션 마스크라고 한다.

```python
def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):

    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []

    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        
        input_id = tokenizer.encode(example, max_length=max_seq_len, pad_to_max_length=True)

       
        padding_count = input_id.count(tokenizer.pad_token_id)
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count

      
        token_type_id = [0] * max_seq_len
        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels
```

- input_id는 워드 임베딩을 위한 문장의 정수 인코딩했다.
- attention_mask는 실제 단어가 위치하면 1, 패딩의 위치에는 0인 시퀀스다.
- token_type_id는 세그먼트 임베딩을 위한 것으로 이번 예제는 문장이 1개이므로 전부 0으로 통일했다.
```python
# example에 댓글, labels에 긍/부정, max_seq_len은 128, tokenizer는 그대로 사용할 것이다.
# 마지막에 return되는 input_ids, attention_masks, token_type_ids가 X데이터로 들어가고  data_labels는 Y로 들어간다.
```

```python
train_X, train_y = convert_examples_to_features(train_data['document'], train_data['label'], max_seq_len=max_seq_len, tokenizer=tokenizer)
test_X, test_y = convert_examples_to_features(test_data['document'], test_data['label'], max_seq_len=max_seq_len, tokenizer=tokenizer)
```
- 이 함수를 이용해 데이터셋을 모델이 학습할 수 있는 형태로 변환했다.
    - x에 본문을, y에는 긍/부정값인 label을 담았다.
```python
train_X[0][0]
train_X[1][0]
train_X[2][0]
```
- 각각 처음부터 input_id, 어텐션마스크, 세그먼트다.

```python
tokenizer.decode(train_X[0][0])
```
- [CLS] 아 더빙.. 진짜 짜증나네요 목소리 [SEP] [PAD] [PAD] [PAD] [PAD] [PAD]...[PAD]로 출력된다.

```python
input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

print('단어에 대한 정수 인코딩 :',input_id)
print('어텐션 마스크 :',attention_mask)
print('세그먼트 인코딩 :',token_type_id)
print('각 인코딩의 길이 :', len(input_id))
print('정수 인코딩 복원 :',tokenizer.decode(input_id))
print('레이블 :',label)
```
- 이 코드를 이용해 각각 코드에 대한 값을 출력했다.

```python
model = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)
max_seq_len = 128
input_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
attention_masks_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
token_type_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)

outputs = model([input_ids_layer, attention_masks_layer, token_type_ids_layer])
```
- 모델을 새로 정의했다.

```python
print(outputs[0])
# 만들고자 하는 모델이 다:다 구조인 겨웅에는 outputs[0]사용한다.
# shape가 none, 128, 768이다 여기서 none의 의미는 배치 크기가 정해져 있지 않다는 의미다.
# 순서대로 (batch_size, max_seq_len, 단어 벡터)다.
# 768차원 벡터가 128개 있다.
print(outputs[1])
# 만들고자 하는 모델이 다:다 구조인 겨웅에는 outputs[0]사용한다.
# shape = (None, 768)
#         (배치크기, 단어벡터)
# 768차원 벡터가 1개 있다.
```
## Bert 다:1구조 모델 생성

```python
class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name): # model_name = klue/bert-base
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='sigmoid',
                                                name='classifier')
# 다중분류시에 출력수와 activation만 변경이 될 수 있다.
    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs[1] # 다:다라면 0번으로 바뀐다.
        prediction = self.classifier(cls_token)

        return prediction
```
- 클래스를 정의했다. 모델의 이름은 klue/bert-base로 전에 사용한 모델과 같은 모델이다.
    - 다:1 구조의 모델이라 sigmoid가 사용된다.
        - 만약 다:다라면 softmax 가 사용.

    - 다:다라면 cls_token이 0번으로 바뀐다.

```python
with strategy.scope():
  model = TFBertForSequenceClassification("klue/bert-base")
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  loss = tf.keras.losses.BinaryCrossentropy()
  model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])
model.fit(train_X, train_y, epochs=2, batch_size=64, validation_split=0.2)
```
- 모델을 정의하고 각각 optimizer와 loss값을 설정해줬다.
    - 모델을 train_x와 train_y 를 두바퀴 돌리고 한번에 64개의 데이터를 받아들이며 검증데이터는 20%로 지정해 훈련시켰다.

```python
results = model.evaluate(test_X, test_y, batch_size=1024)
print("test loss, test acc: ", results)
```
- test값을 예측하고 값을 비교했다.
- ```python
    49/49 [==============================] - 24s 360ms/step - loss: 0.2632 - accuracy: 0.9001
    test loss, test acc:  [0.2632245123386383, 0.9000539779663086]
    ```
```python
sentiment_predict("보던거라 계속 보고 있는데 전개도 느리고 주인공인 은희는 한두컷 나오면서 소극적인 모습에")
```
- -> 0.0072414577   99.28% 확률로 부정 리뷰입니다.

```python
def sentiment_predict(new_sentence):
  input_id = tokenizer.encode(new_sentence, max_length=max_seq_len, pad_to_max_length=True)

  padding_count = input_id.count(tokenizer.pad_token_id)
  attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
  token_type_id = [0] * max_seq_len

  input_ids = np.array([input_id])
  attention_masks = np.array([attention_mask])
  token_type_ids = np.array([token_type_id])

  encoded_input = [input_ids, attention_masks, token_type_ids]
  score = model.predict(encoded_input)[0][0]
  print(score)

  if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
  else:
    print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))
```
- 확률로 리뷰를 긍/부정 분류를 해 출력한다.
