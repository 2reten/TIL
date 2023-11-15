```python
# Bert
# 파인튜닝은 미리 만들어져 있는 모델을 토대로 학습을 시켜 사용자의 목적에 맞게 미세한 조정으로 성능을 높이고 학습시간을 줄이는 것이다.
# 기존의 bert모델에 대한 변경을 최소한으로 하고 classifier로 학습을 하는것을 파인튜닝이라고 한다.
# 세바스찬 루더는 사전 훈련된 언어 모델의 약진을 보며 다음 말을 헀다.
# "사전 훈련된 단어 임베딩이 모든 NLP 실무자의 도구 상자에서 사전훈련된 언어 모델로 대체되는 것은 시간 문제다."
# 임베딩을 하는 방법 2가지는 임베딩 층을 초기화 하여 처음부터 학습을 시키는 방법과 데이터가 부족한 경우에는 사전에 학습된 임베딩 벡터들을 가져와 사용하는 방법이다.
# 하지만 이것의 한계점은 다의어나 동음이의어를 구분하지 못하는 문제점이 있다.
# 이러한 에러사항은 사전 훈련된 언어 모델을 사용하므로서 극복이 가능했고, 이 사전 훈련된 모델이 ELMo 와 BERT다.

## 사전 훈련된 언어 모델
# 사전 훈련된 언어 모델과 사전 훈련된 워드 임베딩의 차이는 워드 임베딩은 문장을 단어 단위로 쪼개서 CBOW 혹은 Skipgram로 중심 단어와 주변 단어들 사이에 어떤 관계가 있는지를 고려해 임베딩을 하는 것이다.
# 즉 사전 훈련된 워드 임베딩은 각각의 단어 벡터들로 변환되고 주변의 단어의 벡터와의 연관성으로 학습을한다.
# 사전 훈련된 언어 모델을 이용해서 완벽히는 아니지만 어느정도 극복이 가능하다(다의어나 동음이의어를 의미)
# 언어 모델은 주어진 텍스트로부터 이전 단어들로부터 다음 단어를 예측하도록 학습한다.
# 즉, 레이블이 필요치 않다.
# 사전 훈련된 워드 임베딩과 마찬가지로 사전 훈련된 언어 모델의 강점은 학습 전 사람이 별도 레이블을 지정해줄 필요가 없다는 것이다.
# 사전 훈련된 언어 모델을 고등학생까지의 과정이라고 생각하면 된다.
# 이후 대학에 진학해 전공을 선택해 전문 분야를 학습하는 것을 파인튜닝이라고 생각할 수 있다.

## ELMo
# ELMo는 순방향 언어 모델과 역방향 언어 모델을 각각 따로 학습시킨다
# 사전 학습을 시킨 뒤에 임베딩 값을 얻는다.
# 이와 같은 임베딩을 함으로써 문맥에 따라 임베딩의 벡터값이 달라진다.
# => 이로 다의어 구분에 대한 문제점을 해결할 수 있었다.
# 인코더 - 디코더 구조의 LSTM을 뛰어넘는 좋은 성능을 얻자, LSTM이 아닌 트랜스포머로 사전 훈련된 언어 모델으 학습하는 시도가 등장했다.
# 트랜스포머의 디코더는 LSTM 언어 모델처럼 순차적으로 이전 단어들로부터 다음 단어를 예측한다.
# Open AI는 트랜스포머 디코더로 총 12개의 층을 쌓은 후에 방대한 텍스트 데이터를 학습시킨 언어모델 GPT-1을 만들었다.
# NLP의 주요 트렌드는 사전 훈련된 언어 모델을 만들고, 이를 특정 태스크에 추가 학습시켜 해당 태스크에서 높은 성능을 얻는것.
# 언어의 문맥이라는 것은 양방향이기 때문에 양방향 학습의 성능이 더 좋다는것이 입증되었다.

## 마스크드 언어 모델
# BERT모델이 학습하는 가장 기본적인 기법
# 입력 텍스트의 단어 집합의 15% 단어를 랜덤으로 마스킹 한 뒤,  그 단어를 예측하도록 하는것이 마스크드 언어 모델이다.
# "나는 [MASK]에 가서 그곳에서 빵과 [MASK]를 샀다" 이런 유형으로 구조가 만들어진다.

##  BERT
# BERT라고 하는 것은 2018년에 구글에서 공개한 사전 훈련된 모델이다.
# 공개 이후 NLP태스크에서 최고 성능을 보여주고 있다.
# BERT는 33억개의 단어를 가지고 학습이 되어졌다.
# BERT가 높은 성능을 가질 수 있던것은 레이블이 없는 방대한 데이터로 사전 훈련된 모델을 가지고
# 레이블이 있는 다른 작업에서 추가 학습을 하고 하이퍼 파라미터를 재조정해 성능이 높게 나오는 기존의 사례들을 참고하였기 때문이다.
# 다른 작업에 대해서 파라미터 재조정을 위한 추가 훈련 과정을 파인 튜닝(Fine-tuning)이라고 한다.
# 사전 학습된 BERT위에 신경망을 한층 추가한다.
# => BERT가 언어 모델 사전 학습 과정에서 얻은 지식을 활용할 수 있으므로 보다 더 좋은 성능을 얻을 수 있다.
# BERT-Base 는 12개의 트랜스포머 인코더를 쌓아올린 구조이고, Large 버전에서는 24개의 트랜스포머 인코더를 쌓아올린 구조다.
# 인코더만을 취해 만든것이 BERT고 디코더를 이용해 만든것이 GPT다.
# 셀프 어텐션 헤드는 다양한 시각으로 생각하면 될 것으로 인지.
# BERT-base는 open AI GPT와 하이퍼 파라미터가 동일한데, BERT연구진이 GPT와 성능을 비교하기 위해 동등한 크기로 설계가 되었다.
# BERT-Large는 BERT의 최대 성능을 보여주기 위해 만들어진 모델이다.
# BERT의 입력은 임베딩 층을 지난 임베딩 벡터들이고, 모든 단어들은 768차원의 임베딩 벡터가 되어 입력으로 사용된다.

# 기계가 모르는 단어가 등장하면 OOV 또는 UNK라고 표현한다.
# 서브워드 분리 작업은 하나의 단어를 더 작은 단위의 의미있는 여러 서브워드들의 조합으로 구성된 경우가 많기 때문에
# 하나의 단어를 여러 서브워드로 분리해서 단어를 인코딩 및 임베딩 하겠다는 의도를 가진 전처리다.

## 바이트 페어 인코딩(BPE)
# OOV문제를 완화하는 알고리즘 중 하나다.데이터 압축 알고리즘이다.
# 연속적으로 가장 많이 등장한 글자의 쌍을 찾아서 하나의 글자로 병합하는 방식을 수행한다.
# 문자열 중 가장 자주 등장하고 있는 바이트의 쌍을 찾는다.
# 자연어 처리에서 BPE는 서브워드 분리 알고리즘이다.
# 글자 단위에서 점차적으로 단어 집합을 만들어 내는 Bottom up 방식의 접근을 사용한다.
# 모든 글자 또는 유니코드 단위로 단어 집합을 만들고, 가장 많이 등장ㅇ하는 유니그램을 하나의 유니그램으로 통합한다.
# BPE의 과정
# 1)딕셔너리 내에 모든 단어를 글자 단위로 분리한다.
# 2) BPE의 특징으로 알고리즘의 동작을 몇 회 반복할 것인지를 사용자가 정한다는 점이다.
# 3) 반복한 값을 토대로 단어의 쌍의 빈도수가 가장 높은 쌍을 통합한다.
# 4) 이것을 지정한 만큼 반복하고 vocabulary에 단어의 쌍 역시 추가한다.
# ex)l, o, w, e, r, n, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest

## 센텐스피스
# 내부 단어 분리를 위한 패키지로 구글에서 제공한다.
# 그 내부에는 BPE와 Unigram Language Model Tokenizer가 들어있다.
# 내부 단어 분리를 하려면 먼저 데이터 단어 토큰화를 먼저 진행한 상태여야 한다.(한국어라면 Konlpy, 영어는 nltk등)

## BERT의 WordPiece
# BERT는 단어보다 더 작은 단위로 쪼개는 서브워드 토크나이저를 사용한다.
# BERT에서 토큰화를 수행하는 방식은 이미 훈련 데이터로부터 만들어진 단어 집합내에 토큰이 단어 집합에 존재한다면 해당 토큰을 분리하지 않고,
# 단어 집하바에 존재하지 않는다면 해단 토큰을 서브워드로 분리한다.
# 단어 집합에 해단 단어가 존재하지 않는 경우 서브워드 토크나이저를 사용하지 않고 그냥 토크나이저 사용시 OOV가 난다.
# 서브워드 토크나이저 사용시 단어가 없다고 OOV 오류를 내는것이 아닌 해당 단어를 더 쪼개려고 시도한다.
# embeddings로 예시를 들어보면, em, ##bed, ##ding, ##s라는 서브워드가 단어 집합 내에 존재하는 경우 em, ##bed, ##ding, ##s라는 서브워드로 분리된다.

## 마스크드 언어 모델
# BERT에서는 마스크드 언어 모델을 사용해 양방향성을 얻었다.
# BERT모델의 사전 훈련 방식은 두가지가 있는데 그중 한가지가 마스크드 언어 모델이고, 다른 한가지는 다음 문장 예측이다.
# BERT는 사전 훈련을 위해서 입력으로 들어가는 입력 텍스트의 15%의 단어를 랜덤으로 마스킹한다.
# 그리고 인공 신경망에게 가려진 단어들을 예측하는 것으로 훈련한다.
```
# IMDB 리뷰 토큰화

```python
pip install sentencepiece
import sentencepiece as spm
import pandas as pd
import urllib.request
import csv
```
- 사용한 모듈들이다.

```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
train_df = pd.read_csv('IMDb_Reviews.csv')
```
- IMDB 리뷰를 저장하고 리뷰를 불러왔다.

```python
with open('imdb_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(train_df['review']))
```
- imdb_review파일이 생성되고 그 내부 내용은 데이터프레임의 review열의 데이터다.

```python
spm.SentencePieceTrainer.Train('--input=imdb_review.txt --model_prefix=imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')
```
- 기존 단어들 + 서브워드로 구성된 Vocaburary를 생성하는 코드다.
- --input 은 생성하고자 하는 단어 집합이 담겨있는 파일이 들억나다.
- vocab_size는 단어 집합의 크기다.
- --model_type는 bpe알고리즘을 적용하겠다는 의미다.
- --max_sentence_length는 문장의 최대 길이를 의미한다.
```python
pd.read_csv("imdb.vocab", sep = "\t", header = None, quoting = csv.QUOTE_NONE)
```
- csv모듈을 이용해서 데이터를 처리할 때 문장에 컴마가 있는경우 이 모듈을 통해서 처리가 가능하다.
- quoting은 묶어줘야 하는 문자열을 처리할 때 사용하는 것이다.

```python
vocab_list = pd.read_csv('imdb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
sp = spm.SentencePieceProcessor()
vocab_file = "imdb.model"
sp.load(vocab_file)
```
- vocab_list에 vocab들을 저장했다.
- 모델을 불러왔다.

```python
lines = [
  "I didn't at all think of it this way.",
  "I have waited a long time for someone to film"
]
sp.encode_as_pieces(lines[0])
sp.encode_as_ids(lines[0])
```
- 문장 => 서브 워드로 변환된다. 분리된 단어들이 출력된다.
- 문장 => 정수 코드로 변환된다. 분리된 단어들의 번호가 출력된다.

```python
sp.PieceToId("▁I")
```
- id출력

```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
naver_df = pd.read_table('ratings.txt')
print(naver_df.isnull().values.any())
```
- Null 값이 존재하는 행 제거했다.
- Null 값이 존재하는지 확인했다.

```python
print('리뷰 개수 :',len(naver_df)) 
```
- 리뷰 개수 : 199992 
```python
with open('naver_review.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(naver_df['document']))
spm.SentencePieceTrainer.Train('--input=naver_review.txt --model_prefix=naver --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')
vocab_list = pd.read_csv('naver.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)
```
- 파일을 새로 저장하고, 그 파일을 읽었다.

```python
sp = spm.SentencePieceProcessor()
vocab_file = "naver.model"
sp.load(vocab_file)
lines = [
  "뭐 이딴 것도 영화냐.",
  "진짜 최고의 영화입니다 ㅋㅋ",
]
print(sp.encode_as_pieces(lines[0]))
print(sp.encode_as_pieces(lines[1]))
```
- 네이버 리뷰로 사전 학습시킨 모델로 lines에 있는 문장을 SentencePieceProcessor로 출력했다.
    - ['▁뭐', '▁이딴', '▁것도', '▁영화냐', '.']
    - ['▁진짜', '▁최고의', '▁영화입니다', '▁ᄏᄏ']
```python
print(sp.DecodePieces(['▁진짜', '▁최고의', '▁영화입니다', '▁ᄏᄏ']))
sp.DecodeIds([54, 200, 821, 85])
```
- 진짜 최고의 영화입니다 ᄏᄏ
- '진짜 원 산~~'

```python
print(sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=str))
print(sp.encode('진짜 최고의 영화입니다 ㅋㅋ', out_type=int))
```
- out_type를 str로 주면 서브 워드로 출력했다.
- out_type를 int로 주면  id로 출력했다.
    - ['▁진짜', '▁최고의', '▁영화입니다', '▁ᄏᄏ']
    - [54, 204, 825, 121]

```python
import pandas as pd
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```
- Bert-base의 토크나이저를 불러왔다.

```python
print(tokenizer.vocab["love"])
print(tokenizer.vocab["loves"])
print(tokenizer.vocab["embeddings"])
```
- 값으로는 각각 2293, 7459가 나왔고 마지막 단어는 내부에  embedding이 없어서 oov오류가 났다.

```python
print(tokenizer.vocab["em"])
print(tokenizer.vocab["##bed"])
print(tokenizer.vocab["##ding"])
print(tokenizer.vocab["##s"])
tokenizer.tokenize("Here is the sentence I want embeddings for.")
```
- 7861, 8270, 4667, 2015가 각각 출력됐고 다음 문장을 bert를 이용해서 토큰화 한 결과가 아래처럼 출력됐다.
    - ['here', 'is', 'the', 'sentence', 'i', 'want', 'em', #bed', '##ding', '##s', 'for', '.']

```python
with open('vocabulary.txt', 'w', encoding = "UTF8") as f:
    for token in tokenizer.vocab.keys():
        f.write(token + '\n')
df = pd.read_fwf('vocabulary.txt', header=None)
```
- 'vocabulary.txt'파일을 만들어 그 안에 값으로 각 단어들을 저장하고 \n 으로 구분한 뒤 데이터프레임으로 열었다.

## BERT

```python
# BERT에서 사용되는 특별토큰
# [PAD] - 0
# [UNK] - 100
# [CLS] - 101
# [SEP] - 102
# [MASK] - 103

# https://huggingface.co/ 사전 학습된 BERT모델 가져오는 사이트
```
```python
pip install transformers
import pandas as pd
from transformers import BertTokenizer
```
- transformers를 다운하고 내부의 berttokenizer와 pandas를 임폴트했다.
```python
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```
- from_pretrained를 이용해서 모델의 토크나이저를 불러왔다.

```python
result = tokenizer.tokenize('Here is the sentence I want embeddings for.')
```
- ['here', 'is', 'the', 'sentence', 'i', 'want', 'em', '##bed', '##ding', '##s', 'for', '.']가 result에 담겨있다.

```python
from transformers import TFBertForMaskedLM
model = TFBertForMaskedLM.from_pretrained('bert-large-uncased')
```
- TFBertForMaskedLM는 BERT를 마스크드 언어 모델 구조로 읽어들인다.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
```
- AutoTokenizer는 bert-large-uncased 모델이 학습될 당시에 사용된 토크나이저가 읽어진다.
    -  A라는 버트토크나이저에서는 '사과'를 40번으로 인코딩
    -  B라는                               50번으로
```python
inputs = tokenizer.encode_plus('Soccer is a really fun [MASK].', add_special_tokens=True, return_tensors='tf')
inputs
```
- inputs의 결과값이다. ```python
{'input_ids': <tf.Tensor: shape=(1, 9), dtype=int32, numpy=
array([[ 101, 4715, 2003, 1037, 2428, 4569,  103, 1012,  102]],
      dtype=int32)>, 'token_type_ids': <tf.Tensor: shape=(1, 9), dtype=int32, numpy=array([[0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>, 'attention_mask': <tf.Tensor: shape=(1, 9), dtype=int32, numpy=array([[1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)>}
      ```
```python
inputs["input_ids"]
inputs["token_type_ids"]
inputs["attention_mask"]
```
- ```python
<tf.Tensor: shape=(1, 9), dtype=int32, numpy=
array([[ 101, 4715, 2003, 1037, 2428, 4569,  103, 1012,  102]],
      dtype=int32)>
      ```
- ```python
<tf.Tensor: shape=(1, 9), dtype=int32, numpy=array([[0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>
```
- ```python
<tf.Tensor: shape=(1, 9), dtype=int32, numpy=array([[1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)>
```

```python
from transformers import FillMaskPipeline
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)
```
- FillMaskPipeline은 마스킹 된 위치에 어떤 단어가 올 확률이 높은가를 측정해 출력한다.

## 한국어 BERT

```python
model = TFBertForMaskedLM.from_pretrained('klue/bert-base', from_pt=True)
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
pip = FillMaskPipeline(model=model, tokenizer=tokenizer)
```
- 영어와 마찬가지 작업이다.

```python
inputs = tokenizer('축구는 정말 재미있는 [MASK]다.', return_tensors='tf')
pip("나는 방금[MASK]를 먹었다.")
```
- pip으로 FillMaskPipeline를 정의해서 모델을 사용한다.

## 다음 문장 예측하기

```python
from transformers import TFBertForNextSentencePrediction
from transformers import AutoTokenizer
model = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```
- 기존과 크게 다른것은 없다.
```python
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
next_sentence = "The sky is blue due to the shorter wavelength of blue light."
encoding = tokenizer(prompt, next_sentence, return_tensors='tf')
```
- 첫 문장과 다음 문장을 정의해주고 두 문장을 토큰화 시켰다.

```python
tokenizer.decode(encoding["input_ids"][0])
```
- ```python
[CLS] in italy, pizza served in formal settings, such as at a restaurant, is presented unsliced. [SEP] the sky is blue due to the shorter wavelength of blue light. [SEP]
```
    - 문장을 디코더로 출력한 결과값이다.

```python
import tensorflow as tf
logits = model(encoding["input_ids"], token_type_ids = encoding["token_type_ids"])[0]
print(tf.keras.layers.Softmax(logits))
soft = tf.keras.layers.Softmax()
res = soft(logits)
tf.math.argmax(res, axis=-1).numpy()[0]
```
-  다음 문장을 예측하는 코드다. token_type_ids이 값을 지정해주지 않는다면 디폴트로 설정되어진 값이 있기에 올바르게 출력이 되지 않는다.
    - 결과 값으로는 1이 출력됐다
        - 1은 이어지지 않는 문장이라는 의미다.