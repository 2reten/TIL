# 연관 분석을 이용한 추천 시스템

```python
"""
자연어: 일상생활에서 사용하는 언어
자연어처리: 음성인식, 요약, 번역, 감성분석, 분류, 질의응답, 챗봇 등
환경구성: 아나콘다(머신러닝, 시각화, 데이터분석, nltk 등) + 텐서플로우, 젠심, 파이토치, konlpy 등

자연어처리에서 해야할일
1) 텍스트 전처리(토큰화, 정제, 어간추출, 불용어제거, 정수인코딩(단어를 수치로), 패딩)
2) 텍스트의 수치표현(BoW, DTM/TDM, TF-IDF)
3) 유사도(문서/단어/문장) - 코사인유사도, 유클리디안거리
4) 머신/딥러닝 모델 생성
"""
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
```python
pip install konlpy
```
```python
from konlpy.tag import Okt
```
- 사용한 모듈들과 사용할 konlpy를 다운하는 코드다.

```python
okt=Okt()
print(okt.morphs(u'단독입찰보다 복수입찰의 경우'))
print(okt.nouns(u'유일하게 항공기 체계 종합개발 경험을 갖고 있는 KAI는'))
print(okt.pos(u'이것도 되나욬ㅋㅋ'))
```
- Otk를 이용해서 형태소를 분리하는 과정이다.
    - ['단독', '입찰', '보다', '복수', '입찰', '의', '경우']
    - ['항공기', '체계', '종합', '개발', '경험']
    - [('이', 'Determiner'), ('것', 'Noun'), ('도', 'Josa'), ('되나욬', 'Noun'), ('ㅋㅋ', 'KoreanParticle')]

```python
from konlpy.tag import *
```
- * 을 이용해서 konlpy.tag내부의 모든 모듈을 다운했다.
```python
okt=Okt()
han=Hannanum()
kkma=Kkma()
```
- 모델을 따로 저장하는 과정이다.
```python
okt.pos("아버지가방에들어가신다")
han.pos("아버지가방에들어가신다")
okt.pos("정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")
han.pos("정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")
kkma.pos("정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")
```
- 각 모델들 별로 다른 값이 나오는것을 알 수 있다.

```python
"""# df("바나나") = 2  
# tf*idf=문서에서 각 단어의 중요도를 나타낸 행렬
# = 각 문서에서 중요 단어가 무엇인지 알고자 함 => 키워드 => 토픽모델링
# tfidf는 단어의 중요도
# tfidf = tf*idf # i 역수 df빈도수
N=문서의 갯수

 N
----  * tf = tfidf
df+1

idf(w) = log(n/(1+df(w))) * tf(w)
log를 적용시키는 이유는 log를 적용시키지 않는다면 문서의 수의 따라서 값이 기하급수적으로 커지기 때문이다.

"""
```
```python
from math import log 
docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]
vocab=list(set(w for doc in docs for w in doc.split()))
vocab.sort()
```
- 코드의 의미는 docs내부의 값 하나하나를 공백을 기준으로 나누고 각 값의 집합으로 하나의 값만을 출력한 뒤 리스트화한 뒤 그 리스트를 정렬했다.
    - ['과일이', '길고', '노란', '먹고', '바나나', '사과', '싶은', '저는', '좋아요']


```python
N = len(docs) 

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df+1))
  
def tfidf(t, d):
    return tf(t,d)* idf(t)
result = []

# 각 문서에 대해서 아래 연산을 반복
for i in range(N):
    result.append([]) # [  []      ]
    d = docs[i] # '먹고 싶은 사과'
    for j in range(len(vocab)): #9번반복
        t = vocab[j] #'과일이'
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = vocab)
```
- tfidf값을 구하는 각각의 함수를 미리 만들어두고 아래 코드로는 tf의 값을 구하는 코드를 만들고 그 값을 데이터로 데이터프레임화 했다.

```python
from sklearn.feature_extraction.text import CountVectorizer
```
- countvectorizer를 구현하는 코드다. 이 코드를 이용해서 docs를 출력하면 데이터프레임이 아닌 같은 값을 지니는 array로 출력된다.
    - ```python
    array([[0, 0, 0, 1, 0, 1, 1, 0, 0],
       [0, 0, 0, 1, 1, 0, 1, 0, 0],
       [0, 1, 1, 0, 2, 0, 0, 0, 0],
       [1, 0, 0, 0, 0, 0, 0, 1, 1]])
    ```
```python
vec.vocabulary_ 
```
- {'먹고': 3,
 '싶은': 6,
 '사과': 5,
 '바나나': 4,
 '길고': 1,
 '노란': 2,
 '저는': 7,
 '과일이': 0,
 '좋아요': 8} 의 값을 가진다. 벨류 부분은 인덱스 값이다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer #쉽게 tf_idf행렬 구할수 있는 라이브러리
```
```python
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns=["IDF"])
idf_

#idf(역수이므로) 값이 클수록 흔치않은 단어
```
- 이 코드는 아까 만든 함수로 idf의 값을 출력하는 코드다.

```python
result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t,d))

tfidf_ = pd.DataFrame(result, columns = vocab)
tfidf_ #값이 클수록 중요도가높다
```
- 빈 리스트를 만들고 d를 docs의 각 문장을 저장
    - 그리고 이중for문을 이용해서 vocab의 값을 각각 tfidf함수를 이용해서 그 값을 만들어진 빈 리스트에 넣는 구조다.

```python
pip install tensorflow
pip install gensim
pip install nltk
```
```python
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import nltk
nltk.download('punkt')
```
- 사용할 모듈의 다운로드와 import하는 과정이다.

```python
"""
토큰? 자연어처리 작업을 수행하는 기본단위, 
일반적으로 단어,문장(문단을 문장단위로나눈), 문단(문서를 문단단위로나눈), 문자
토큰화? 주어진 코퍼스를 토큰 단위로 나누는 작업
자연어 -> 토큰화 -> 세부 작업
"""
```
```python
print('단어 토큰화1 :',word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
print('단어 토큰화2 :',WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
print('단어 토큰화3 :',text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```
- 각각 다른 함수를 써서 다른 값이 출력되는걸 알 수 있다.
    - 단어 토큰화1 : ['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
    - 단어 토큰화2 : ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
    - 단어 토큰화3 : ["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
```python
from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문장 토큰화1 :',sent_tokenize(text))
text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화2 :',sent_tokenize(text))
```
- 문장 단위로 토큰화를 하는 함수다.
    - 문장 토큰화1 : ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 'He looked about, to make sure no one was near.']
    - 문장 토큰화2 : ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']

```python
from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 품사 태깅 :',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('OKT 명사 추출 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```
- 출력값이다. 
    - OKT 형태소 분석 : ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']
OKT 품사 태깅 : [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
OKT 명사 추출 : ['코딩', '당신', '연휴', '여행']
```python
print('꼬꼬마 형태소 분석 :',kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 품사 태깅 :',kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print('꼬꼬마 명사 추출 :',kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
```
- 꼬꼬마 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
꼬꼬마 품사 태깅 : [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
꼬꼬마 명사 추출 : ['코딩', '당신', '연휴', '여행']

```python
import re
text = "I was wondering if anyone out there could enlighten me on this car."

shortword = re.compile(r'\W*\b\w{1,2}\b')
print(shortword.sub('', text))
```
-  길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제

```python
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
# are, is , am =>be(표제어)
lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']

print('표제어 추출 전 :',words)
print('표제어 추출 후 :',[lemmatizer.lemmatize(word) for word in words])
```
- 표제어(동의어)를 처리하는 과정이다.
```python
lemmatizer.lemmatize("is", "v")
lemmatizer.lemmatize("are", "v")
lemmatizer.lemmatize("watched", "v")
lemmatizer.lemmatize("watching", "v")
```
- 출력값으로는 모두 be와 watch가 출력된다.

```python
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()
words = ['formalize', 'allowance', 'electricical']
print('어간 추출 후 :',[stemmer.stem(word) for word in words])
print('어간 추출 후 :',[lancaster_stemmer.stem(word) for word in words])
```
- 어간을 추출하는 두가지의 함수다.
    - 첫번째 함수는 어간 추출 후 : ['formal', 'allow', 'electric'] 값을 출력했다.
    - 두번째 함수는 어간 추출 후 : ['form', 'allow', 'elect'] 값을 출력했다.

```python
from nltk.corpus import stopwords
```
- 불용어를 처리하기 위한 모듈을 import했다.

```python
example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(example)

result = []
for word in word_tokens: 
    if word not in stop_words: 
        result.append(word) 

print('불용어 제거 전 :',word_tokens) 
print('불용어 제거 후 :',result)
```
- 불용어를 제거하는 코드다.
    - 불용어 제거 전 : ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
불용어 제거 후 : ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']

```python
example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는"

stop_words = set(stop_words.split(' '))
word_tokens = okt.morphs(example)

result = [word for word in word_tokens if not word in stop_words]

print('불용어 제거 전 :',word_tokens) 
print('불용어 제거 후 :',result)
```
- 이 코드는 한국어의 불용어를 제거하는 방식이다.
    - 미리 stop_words에 코드를 준 뒤 그 값을 제외하고 출력하는 방식이다. 방식은 같으나 영어는 자체값을 사용하고 한국어는 따로 지정을 해야한다.

```python
from nltk.corpus import wordnet
```
- 동의어를 출력해주는 코드다.

```python
"""
1. TF-IDF기반 영화 추천시스템
2. 실행과정
즐겁게 봤던 영화 제목을 입력하세요
당신에게 추천하고 싶은 영화 제목은 아래와 같습니다.
1)Jumanji
2)Grumpier Old Men
...
10)

3. 개발방법
1) overview 열 추출 -> 단어 전처리(불용어제거, 대소문자, 단어 통일(wordnet), 특수문자 처리, 정규식) 
-> 코퍼스
2) tf-idf 행렬 생성
3) 코사인유사도를 이용한 영화 추천

4. 데이터셋
- 5000편 영화

5. 카페 제출
"""
```
- 오늘의 과제다.
```python
data = pd.read_csv("archive/movies_metadata.csv")
data = data.head(5000)
from tensorflow.keras.preprocessing.text import text_to_word_sequence
```
- tensorflow의 내부 함수를 import 했다. 또 어제의 데이터를 다시 불러왔다.
```python
overview=data['overview']
overview=overview.astype(str)
```
- overview열을 문자열 형태로 변환했다.

```python
result = [] #모든 결과물을 저장할 리스트
for i in range(len(overview)):
    example=overview[i]
    stop_words = set(stopwords.words('english')) 

    word_tokens = text_to_word_sequence(example)
    result_movie = [x for x in word_tokens if x not in stop_words]
    result.append(result_movie)
```
- overview의 길이 만큼, 즉 모든 overview의 값을 각각 한번씩 example에 저장하고, 그 값을 토큰화 해서 word_tokens라고 저장한 뒤, result_movie에 불용어에 없는 토큰들의 리스트를 만들고 그 리스트를 result에 추가하는 코드다.
