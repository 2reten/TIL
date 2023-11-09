# 임베딩

```python
# http://www.gutenberg.org -> 저작권 없는 소설
# 희소표현은 원핫인코딩
# 희소표현은 공간적 낭비가 매우 심하다. -> 많이 권장되는 방법은 아님
# 일반적으로는 밀집표현은 많이 사용한다.
# 0과 1만 가진 값이 아닌 실수값을 가지게 된다.
# 메모리의 관점에서 희소표현보다 밀집표현이 낭비나 속도등이 우세하다.
워드 임베딩
# 단어를 밀집 벡터의 형태로 표현하는 방법이다.
# 이 밀집 벡터를 워드 임베딩 과정을 통해 나온 결과라고 하여 임베딩 벡터라고도 불린다.
워드투벡터
# 일반적인 원 핫 벡터는 단어 간 유사도를 계산할 수 없다. -> 축이 늘어날 때마다 축간의 각은 90도이기 때문이다.
# 워드투벡터는 단어간의 유사도를 반영할 수 있다.
# 벡터 공간에 각각의 단어들을 표현하게 된다면 단어들 간 벡터 연산을 통해 추론까지도 가능해질 수 있다.
# 연산이 가능한 이유는 유사도를 반영하고 있기 때문이다.
# 워드투벡터는 단어들을 임베딩 하는 것인데 단순히 수치화 하는것이 아니라 유사도 연산이 가능하게 만들어준다.
# 주변단어들을 입력했을 때 중심단어를 찾아내도록 훈련 시키는것은 CBOW라고 한다
# 윈도우 크기가 모델의 성능을 바꿀 정도로 중요하다
# 윈도우를 옆으로 움직이면서 주변단어와 중심단어의 선택을 변경해가며 학습을 위한 데이터셋을 만드는 방법을 슬라이딩 윈도우라고 한다.
# 워드투벡터에서 모든 값은 원핫 벡터로 표현이 되어야 한다.
# CBOW는 중심단어를 기준으로 앞 뒤로 윈도우의 크기만큼의 단어들을 참조하고, 그 단어들의 원핫 벡터를 입력 레이어에 입력한다.
# 출력 벡터에는 예측하고자 하는 값이 원핫 벡터로 들어가 있다.
# 히든 계층은 projecction layer라고 부른다.
# 히든 계층은 룩업 테이블이라는 연산을 담당하는 층이다. 따로 활성화 함수는 존재하지 않는다.
# 히든 계층의 크기가 M이라면 CBOW에서 히든 계층의 크기 M은 임베딩을 하고 난 벡터의 차원이된다.
# CBOW를 수행하고 나서 얻는 각 단엉의 임베딩 벡터의 차원은 5가 된다.
# 입력 계층과 히든계층 사이의 가중치W는 단어 집합의 크기(V)와 임베딩 차원(M)의 곱이다.
# 히든계층에서 출력층 사이의 가중치는 M * V다.
# 둘은 서로 다른 행렬 ex) (7, 5), (5, 7)
# CBOW는 softmax를 지나면서 백터의 각각의 원소들의 값은 0과 1사이의 실수로 총 합은 1이 된다.,
# 다중 클래스 분류 문제를 위한 일종의 스코어 벡터다. 
```
```python
import re
import urllib.request 
import zipfile 
from lxml import etree 
from nltk.tokenize import word_tokenize, sent_tokenize 
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
```
- urllib.request는 웹에 접속해서 텍스트, 리소스 등을 가져올 때 사용한다.
- zipfile은 압축 파일도 바로 사용할 수 있게 만들어주는 모듈이다.
- lxml의 etree는 분석을 하기 위해 가져온 모듈이다.
- Word2Vec는 단어를 벡터화 시키는 모듈이다.

```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")
targetXML =open('ted_en-20160408.xml', 'r', encoding = "UTF8")
target_text=etree.parse(targetXML)
parse_text = "\n".join(target_text.xpath("//content/text()"))
```
- 데이터를 페이지에서 가져와 저장한 뒤에 그 파일을 UTF8로 읽어서 불러왔다.
    - xpath로 content내에 있는 텍스트 부분만 가져왔다.
```python
content_text = re.sub(r'\([^)]*\)', '', parse_text)
sent_text = sent_tokenize(content_text)
normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower()) # 영어와 숫자를 제외하고는 모두 공백으로 치환한다.
    normalized_text.append(tokens)
```
- 소괄호로 묶인 내용은 모두 제거시켰다.
    - 그 데이터를 토큰화 시켰고, 토큰들을 모두 소문자로 변환, 영어와 숫자를 제외하고는 모두 공백으로 치환했다.

```python
model = Word2Vec(sentences = result, vector_size = 100, window = 5, min_count = 5, sg = 0, workers = 4) 
model.wv.most_similar("phone")
[('card', 0.7665501832962036),
 ('phones', 0.6980592608451843),
 ('telephone', 0.6969967484474182),
 ('facebook', 0.6951161623001099),
 ('car', 0.6372690200805664),
 ('smartphone', 0.6345928311347961),
 ('mobile', 0.6341112852096558),
 ('camera', 0.6312768459320068),
 ('cell', 0.6159112453460693),
 ('iphone', 0.6145120859146118)]
```
- 모델을 만들었다. vector_size는 히든 계층의 차원수를 의미하고 window는 좌,우로 5개씩 포함해서 연관성을 찾는다는 뜻, min_count는 최소 5번 이상 나온 값들에 한해서 벡터화 한다는 의미이고 sg는 skipgram인데 이것을 0으로 주면 skipgram이 아닌 CBOW가 적용된다.
    - Word2vec 모델 학습 과정에서 병렬 처리를 위해 사용되는 작업자(worker) 수를 나타낸다.


```python
model.wv.save_word2vec_format('eng_w2v')
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v")
loaded_model.most_similar("money")
[('attention', 0.6272070407867432),
 ('cash', 0.5877038240432739),
 ('credit', 0.577326238155365),
 ('budget', 0.5637192130088806),
 ('dollars', 0.5593224167823792),
 ('funding', 0.5501046776771545),
 ('effort', 0.5496085286140442),
 ('revenue', 0.5406341552734375),
 ('paid', 0.5278459191322327),
 ('medication', 0.5270339250564575)]
```
- 만든 모델을 저장하고 그 모델을 불러와 불러온 모델로 값을 확인했다.

```python
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
```
```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
train_data = pd.read_table('ratings.txt')
train_data = train_data.dropna(how = 'any')
```
- 영화 댓글 데이터를 가져왔다.
    - 그 데이터를 불러와 결측값 행을 제거했다.
```python
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") 
```
- ㄱ~ㅎ, ㅏ~ㅣ, 가~힣 즉 모든 한글 문자와 공백만을 제외하고 모든 값을 제거하라는 의미다.

```python
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()
```
- 불용어를 지정하고 형태소 분석기를 사용하기 위해서 정의했다.

```python
okt.morphs(train_data["document"][0])
okt.morphs(train_data["document"][0], stem = True)
```
- 둘의 차이는 그냥 분리를 하는가 아니면 어간을 추출해 분리를 하는가로 갈린다.
    - ['어릴', '때', '보고', '지금', '다시', '봐도', '재밌어요', 'ㅋㅋ'] 와 ['어리다', '때', '보고', '지금', '다시', '보다', '재밌다', 'ㅋㅋ']
```python
pip install tqdm
from tqdm import tqdm
```
- 진행도를 알 수 있는 모듈이다.

```python
tokenized_data = []
for sentence in tqdm(train_data['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)
```
- 불용어를 제거하고 토큰화를 하는 작업이다.
```python
from gensim.models import Word2Vec
model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)
```
- 차원을 100 차원 주변값은 5, 최소 카운트도 5, 그리고 workers도 4로 이전과 같이 줬다.

```python
model.wv.vectors.shape # (16477, 100)
print(model.wv.most_similar("쓰레기"))
print(model.wv.most_similar("사이코"))
```
- [('삼류', 0.65082848072052), ('졸작', 0.5856423377990723), ('거지같다', 0.5514147877693176), ('최악', 0.5295501947402954), ('똥', 0.5219247341156006), ('류', 0.508537769317627), ('엉터리', 0.5040706396102905), ('호구', 0.4958325922489166), ('에로영화', 0.4956127405166626), ('베스트', 0.49056509137153625)]
- [('콩가루', 0.7913090586662292), ('하드보일드', 0.782907247543335), ('조장', 0.7816380262374878), ('이사장', 0.7737765312194824), ('부추기다', 0.7626500725746155), ('리즘', 0.7586222887039185), ('베스', 0.755420982837677), ('하류', 0.7523224353790283), ('악인', 0.7519126534461975), ('오락가락', 0.7518628835678101)]

```python
import gensim
import urllib.request
import gensim.downloader as api

wv = api.load('word2vec-google-news-300')
```
- api를 불러와 wv에 저장했다.

```python
wv.vectors.shape #3백만개 단어, 300차원
print(wv.similarity('this', 'is'))
print(wv.similarity('post', 'book'))
```
- 0.40797037
0.057204388

```python
model.wv.save_word2vec_format("kor_w2v")
!python -m gensim.scripts.word2vec2tensor --input kor_w2v --output kor_w2v
```
- 모델을 저장하고 그 모델을 tensor와 metadata로 나눠서 저장하는 코드다.
    - https://projector.tensorflow.org/에서 시각화 가능하다.