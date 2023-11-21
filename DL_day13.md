```python
# 케라스를 이용한 태깅 작업
# https://wikidocs.net/33805 이 사이트의 12강의 초반부는 전처리의 과정이 담겨있다.
# 개체명 인식
# 코퍼스로 부터 각 개체의 유형을 인식한다.
# 이를  사용하면 코퍼스로부터 어떤 단어가 사람, 장소, 조직 등을 의미하는 단어인지 찾을 수 있다.
# NER(개체명 인식) - "유정이는 2018년에 골드만삭스에 입사했다."
# -> 유정 - 사람, 2018년 - 시간, 골드만삭스 - 조직과 같은 형태로 각 개체명 인식을 수행한다.
# 이는 지도학습 기반의 알고르즘을 바탕으로 NER시스템을 만들 수 있다.
# 개체명 인식의 BIO 표현 이해하기 - BIO태깅 기법
# 정보를 텍스트로부터 추출할 때 자주 사용하는 기법이다.
# 개체명 인식은 챗봇 등에서 필요한 주요 전처리 작업이면서 그 자체로도 까다로운 작업이다.
# BIO표현 - Begin Inside Outside의 약자
# B는 개체명이 시작되는 부분에서 출력되고, I는 B나 I다음에 나온다. O는 개체명이 아닌 부분을 표기한다.
# 해리포터 보러가자 => 해B,리I,포I,터I,보O,러O,가O,자O
# 이처럼 영화 제목에 대해서만 개체명을 인식하면 영화 제목이 시작되는 글자인 "해"는 B가 영화 제목이 끝나는 부분까지 "I"가 제목이 아닌 부분에는 "O"가 사용된다.
# 개체명 인식이라는 것은 한 종류의 개체에 대해서만 언급하는 것이 아니라 여러 종류에 언급이 가능하다.
# "해리포러 보러 메가박스 가자" => 해B,리I,포I,터I,보O러O,메B가I박I스I가O자O로 각각 영화제목과 영화관의 이름을 개체로 인식해 분리했다.
# 개체명 인식 데이터 이해하기
# CONLL2003은 개체명 인식을 위한 전통적인 영어 데이터셋이다.
# 여기서 데이터의 형식은 [단어], [품사 태깅], [청크 태깅], [개체명 태깅]의 형식으로 되어있다.
# 개체명 태깅의 경우 LOC는 location, ORG는 organization, PER은 person, MISC는 miscellaneous를 의미한다.
# 개체명 인식은 한 문장이 끝난다면 문장과 문장 사이에 공백으로 분리한다.
# 데이터가 구성되면 정수 인코딩 과정을 거친 후, 모든 데이터의 길이를 동일하게 맞추는 패딩 작업을 거친 후에 딥러닝 모델의 입력으로 사용된다.
# 양방향으로 학습하는 이유는 다음 단어와 이전 단어와의 관계 뿐 아니아 이전 단어와 다음 단어와의 관계도 학습을 시키는 편이 정확도가 높게 나오기 때문이다.
# 토픽 모델링
# 중요한 단어를 추출하거나 중요한 문장을 추출하는 등의 문서를 요약하는 모델이라고 할 수 있다.
# 자연어 처리에서 토픽 모델링은 주제어를 뽑거나 요약본을 만드는 것이라고 할 수 있다.
# 텍스트가 가지고 있는 본래의 의미를 최대한 유지하면서 텍스트의 내용을 간략히 줄이는 것이다.
# -> 중요한 단어들, 문장들만 뽑아 문장을 만드는 것이다.

# 문서 요약은 크게 3가지의 과정으로 정리가 가능하다.
# 1) I(interpretation) : 문서(텍스트)를 해석하고 컴퓨터가 이해할 수 있도록 표현하는 것이다.
# 2) T(transformation) : 요약문으로 표현할 수 있도록 본래의 문장을 가공, 변형하는 것을 말한다.
# 3) G(Generation) : 생성된 요약문들로 부터 최종 요약문을 생성하는 것이다.

# 1개의 문서를 요약할 수도 있고 여러개의 문서를 요약할 수도 있다.
# 또 일반 문서를 요약할 수도 있고 전문분야의 문서를 요약할 수도 있기 때문에 분류의 방법을 고민해볼 필요가 있다.
# 요약문에 대한 문장의 길이를 유동적으로 할 것인가 아니면 고정적으로 할 것인가도 고민해야한다.
# 편파성 - 문서를 요약할 때 편파적인 문서를 만들어낼 가능성도 있기 때문에 이 역시 고려해야 한다.
# 추출 형식은 중요한 단어들을 선별해 그 단어들로 문서를 요약하는 것이다. -> 중요 단어 선별
# 추상 형식은 전체적은 문서를 요약해서 문장을 표현하는 것이다. -> 문서 요약

# LSA는 문서 안에서 빈번하게 등장하는 단어들을 이용해 수치화 하는 방법이다.
# 이는 단어의 의미를 고려하지 못한다는 단점이 있다.
# SVD - 특이값 분해
# 실수 공강 벡터에서 행렬A(m,n)일때, 3개의 행렬 곱으로 분해하는 것을 말한다.
# mm의 직교행렬, nn의 직교행렬, mn의 대각행렬이 3개의 행렬이다.
# 직교행렬은 자신과 자신의 전치행렬의 곱 또는 이를 반대로 곱한 결과가 단위 행렬이 되는 행렬을 말한다.
# 대각행렬은 주대각선을 제외한 곳의 원소가 모두 0인 행렬을 의미한다.
# SVD로 나온 대각 행렬의 대각 원소의 값을 행렬 A의 특이값이라고 표현된다.
# 전치 행렬 - transpose 행렬 row와 columns를 바꿔 반사 대칭을 하여 얻는 행렬이다.
# 단위 행렬 - identity 행렬 주대각요소의 원소가 모두 1이며 나머지 원소의 요소는 모두 0인 정사각 행렬을 의미한다.
# 역 행렬 - 역 행렬을 행렬 A를 어떤 행렬을 곱했을 때, 단위 행렬이 나온다면 여기서 어떤 행렬은 A의 역행렬이라고 한다.
# 직교 행렬 - orthogonal matrix는
# 대각 행렬 - 대각 요소값이 0이 아니고 나머지가 0인 행렬을 대각행렬이라고 부른다.
# 직사각 행렬이라면 1,1 부터 순차적으로 a의 값을 가지고 행 혹은 열이 없다면 같은 행과 열이 있는 부분까지 a의 값을 가진다.
# SVD를 통해서 나온 대각행렬의 주대각원소를 우리는 행렬 A의 특이값이라고 한다.
# 이때 특이값은 내림차순 정렬되어져있다.

# SVD는 풀SVD와 절단된 SVD로 나뉜다.
# LSA의 경우 풀 SVD에서 나온 3개의 행렬에서 일부 벡터들을 삭제시킨 절단된 SVD라고 부른다.
# 절단된 SVD는 대각행렬의 원소 값 중에서 상위값 t개만 남게된다.
# 절단된 SVD를 수행하면 값의 손실이 일어나므로 기존의 행렬A를 복구할 수 없다.
# U행렬과 V행렬도 t열까지만 남기면 된다.
# 이때 t를 작게 잡으면 노이즈(극단값)제거가 잘 되고, 크게 잡으면 노이즈 제거는 잘 되지 않지만 행렬로부터 다양한 의미를 가져갈 수 있다는 장점이 있다.
# 일부 벡터들을 삭제하는 것을 데이터의 차원을 줄인다고도 말하는데, 이때 계산 비용이 낮아지는 효과를 얻을 수 있다.
# 계산 비용이 낮아지는 것 외에도 중요하지 않은 정보를 삭제하는 효과를 가지고 있는데 영상처리 분야에서는 노이즈 제거한다는 의미를 갖는다.
# 자연어 처리 분야에서는 설명력이 낮은 정보를 삭제하고 설명력이 높은 정보만을 남긴다는 의미다.
# 잠재 의미 분석
# 기존의 DTM이나 DTM에 단어의 중요도에 따른 가중치를 주었던 TF-IDF 행렬은 단어의 의미를 전혀 고려하지 못한다는 단점을 가지고 있다.
# LSA는 기본적으로 DTM이나 TF-IDF 행렬에 절단된 SVD(truncated SVD)를 사용하여 차원을 축소시키고, 단어들의 잠재적인 의미를 끌어낸다는 아이디어를 갖고 있다.
# LSA의 장점은 쉽고  빠르게 구현이 가능할 뿐 아니라 단어의 잠재적인 의미를 이끌어낼 수 있어서 문서의 유사도 계산 등에서 좋은 성능을 보여준다는 장점이있다.
# LSA의 단점으로는 SVD의 특성상 이미 계산된 LSA에 새로운 데이터를 추가하여 계산하려고하면 보통 처음부터 다시 계산해야 한다.
# 즉 업데이트가 어렵다는 단점이 있다.
# 잠재 디리클레 할당
# 검색 엔진, 고객 민원 시스템등과 같이 문서의 주제를 알아내는 일이 중요한 곳에서 사용된다.
# LDA는 문서들은 토필들의 혼합으로 구성되어져 있고, 토픽들은 확률 분포에 기반하여 단어들을 생성한다고 가정한다.
```

```python
import numpy as np
A = np.array([[0,0,0,1,0,1,1,0,0],[0,0,0,1,1,0,1,0,0],[0,1,1,0,2,0,0,0,0],[1,0,0,0,0,0,0,1,1]])
```
- 4개의 문서에 9개의 단어가 있고 내부 값은 각각의 빈도수다.

```python
U, s, VT = np.linalg.svd(A, full_matrices = True)
```
- ```python
    u =  [[-0.24  0.75  0.   -0.62]
    [-0.51  0.44 -0.    0.74]
    [-0.83 -0.49 -0.   -0.27]
    [-0.   -0.    1.    0.  ]]
    s = [2.69 2.05 1.73 0.77]
    ```

```python
import numpy as np
from numpy.linalg import svd
np.random.seed(121)
a = np.random.randn(4,4)
print(np.round(a, 3))
```
- 4x4 랜덤배열을 만들었다.
```python
np.diag(Sigma)
np.dot(np.dot(U,np.diag(Sigma)),Vt)
```
- 대각행렬과 u, vt를 행렬곱을 연산하면 원래의 값과 유사한 값이 나온다.

```python
np.random.seed(121)
matrix = np.random.random((6, 6))
print('원본 행렬:\n',matrix)
U, Sigma, Vt = svd(matrix, full_matrices=False)
print('\n분해 행렬 차원:',U.shape, Sigma.shape, Vt.shape)
print('\nSigma값 행렬:', Sigma) 
```
- 원본 행렬을 출력하고, SVD를 적용할 경우 U, Sigma, Vt 의 차원 확인하는 코드다.
    - 특이값 행렬 -> 잠재되어져 있는 특성이다.

```python
num_components = 4
U_tr, Sigma_tr, Vt_tr = svd(matrix, num_components)
print('\nTruncated SVD 분해 행렬 차원:',U_tr.shape, Sigma_tr.shape, Vt_tr.shape)
print('\nTruncated SVD Sigma값 행렬:', Sigma_tr)
matrix_tr = np.dot(np.dot(U_tr,np.diag(Sigma_tr)), Vt_tr)

print('\nTruncated SVD로 분해 후 복원 행렬:\n', matrix_tr)
```
- output of TruncatedSVD

```python
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd

np.random.seed(121)
matrix = np.random.random((6, 6))
print('원본 행렬:\n',matrix)
U, Sigma, Vt = svd(matrix, full_matrices=False)
print('\n분해 행렬 차원:',U.shape, Sigma.shape, Vt.shape)
print('\nSigma값 행렬:', Sigma)


num_components = 4
U_tr, Sigma_tr, Vt_tr = svds(matrix, k=num_components)
print('\nTruncated SVD 분해 행렬 차원:',U_tr.shape, Sigma_tr.shape, Vt_tr.shape)
print('\nTruncated SVD Sigma값 행렬:', Sigma_tr)
matrix_tr = np.dot(np.dot(U_tr,np.diag(Sigma_tr)), Vt_tr)  # output of TruncatedSVD

print('\nTruncated SVD로 분해 후 복원 행렬:\n', matrix_tr)
```
- 지금까지의 코드를 정리한 코드다.
```python
import pandas as pd
import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
data = pd.read_csv('/content/drive/MyDrive/abcnews-date-text.csv', error_bad_lines=False)
```
- error_bad_lines를 이용해서 오류가 발생하는 줄을 제외하고 출력했다.

```python
text = data[['headline_text']]
```
- headline_text값을 text로 저장했다.

```python
text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis=1)
stop = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop)])
```
- headline값을 불용어를 제외하고 다시 저장했다.

```python
text['headline_text'] = text['headline_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos='v') for word in x])
tokenized_doc = text['headline_text'].apply(lambda x: [word for word in x if len(word) > 3])
```
- 3인칭 단수 표현 -> 1인칭으로 변환하는 코드와 길이가 3 이하인 단어를 제거하는 코드다.

```python
detokenized_doc = []
for i in range(len(text)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
text['headline_text'] = detokenized_doc
```
- 역토큰화로 토큰화 작업을 되돌린 뒤 그 값을 text['headline_text']에 다시 저장했다.

```python
vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000)
X = vectorizer.fit_transform(text['headline_text'])
```
- 상위 1,000개의 단어만을 보존시켰다.

```python
from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=777, max_iter=1)
```
- n_components는 토픽의 개수를 의미한다.
- learning_method는 online과 batch의 값을 가지는데 online은 하나하나의 값을 모두 돌기에 batch가 더 빠르다.
- max_iter는 epochs라고 생각하면 된다.
- 모든 프로세스를 사용하고 싶다면 n-jobs를 -1로 지정해주면 된다.
```python
lda_top = lda_model.fit_transform(X)
terms = vectorizer.get_feature_names_out()
```

```python
def get_topics(comp, fn, n=10):
  for idx, topic in enumerate(comp):
    print("Topic %d:" % (idx+1), [(fn[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(lda_model.components_, terms)
```
- 10개의 토픽으로 나눠 각 토픽의 주제를 예측할 수 있는 코드가 완성됐다.

```python
# NLI : 두 문장 유사성 -> 두 문서의 수반, 모순, 중립의 관계를 출력시킨다.
```
```python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import urllib.request
from sklearn import preprocessing
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
```

```python
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/multinli.train.ko.tsv", filename="multinli.train.ko.tsv")
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/snli_1.0_train.ko.tsv", filename="snli_1.0_train.ko.tsv")
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/xnli.dev.ko.tsv", filename="xnli.dev.ko.tsv")
urllib.request.urlretrieve("https://raw.githubusercontent.com/kakaobrain/KorNLUDatasets/master/KorNLI/xnli.test.ko.tsv", filename="xnli.test.ko.tsv")
```
- 각각 훈련데이터, 검증데이터, 테스트 데이터를 저장했다.

```python
train_snli = pd.read_csv("snli_1.0_train.ko.tsv", sep='\t', quoting=3)
train_xnli = pd.read_csv("multinli.train.ko.tsv", sep='\t', quoting=3)
val_data = pd.read_csv("xnli.dev.ko.tsv", sep='\t', quoting=3)
test_data = pd.read_csv("xnli.test.ko.tsv", sep='\t', quoting=3)

train_data = train_snli.append(train_xnli)
train_data = train_data.sample(frac=1)
```
- train데이터를 결합한 뒤 섞었다.

```python
def drop_na_and_duplciates(df):
  df = df.dropna()
  df = df.drop_duplicates()
  df = df.reset_index(drop=True)
  return df

train_data = drop_na_and_duplciates(train_data)
val_data = drop_na_and_duplciates(val_data)
test_data = drop_na_and_duplciates(test_data)
```

- 각 데이터의 결측값과 중복 샘플을 제거했다.

```python
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
max_seq_len = 128
sent1 = train_data['sentence1'].iloc[0]
sent2 = train_data['sentence2'].iloc[0]
```
- 각 데이터의 첫 문장을 저장했다.

```python
encoding_result = tokenizer.encode_plus(sent1, sent2, max_length=max_seq_len, pad_to_max_length=True)
```
- 인코딩작업을 해줬다. 사전 학습된 bert에 값을 찾아서 그 값과 빈 값은 모두 0으로 채웠다.
```python
def convert_examples_to_features(sent_list1, sent_list2, max_seq_len, tokenizer):

    input_ids, attention_masks, token_type_ids = [], [], []

    for sent1, sent2 in tqdm(zip(sent_list1, sent_list2), total=len(sent_list1)):
        encoding_result = tokenizer.encode_plus(sent1, sent2, max_length=max_seq_len, pad_to_max_length=True)

        input_ids.append(encoding_result['input_ids'])
        attention_masks.append(encoding_result['attention_mask'])
        token_type_ids.append(encoding_result['token_type_ids'])

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    return (input_ids, attention_masks, token_type_ids)
X_train = convert_examples_to_features(train_data['sentence1'], train_data['sentence2'], max_seq_len=max_seq_len, tokenizer=tokenizer)
```
- 모든 값에 대해서 인코딩을 해주는 작업이다.

```python
input_id = X_train[0][0]
attention_mask = X_train[1][0]
token_type_id = X_train[2][0]

print('단어에 대한 정수 인코딩 :',input_id)
print('어텐션 마스크 :',attention_mask)
print('세그먼트 인코딩 :',token_type_id)
print('각 인코딩의 길이 :', len(input_id))
```
- ```python
    단어에 대한 정수 인코딩 : [    2   553  2565  2079  5685  2470 29822  2170  2259  5725  2530  2052
  1415  2359  3683    16  1039  2073  3611  7285   636  2116  6422  2118
  1380  2259  4000   575  2069  5589  1902  2069   904  6509   636  2259
  4114  2069  6968  2205  2259   575  2069 15667  2062    18     3   636
  4801   921  2073   636  3634  3910  2138  5695  2075  2318  1902  2062
    18     3     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0]
    어텐션 마스크 : [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    세그먼트 인코딩 : [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    각 인코딩의 길이 : 128
    ```
```python
X_val = convert_examples_to_features(val_data['sentence1'], val_data['sentence2'], max_seq_len=max_seq_len, tokenizer=tokenizer)
X_test = convert_examples_to_features(test_data['sentence1'], test_data['sentence2'], max_seq_len=max_seq_len, tokenizer=tokenizer)
```
- 검증 데이터와 테스트 데이터에도 같은 작업을 해줬다.

```python
train_label = train_data['gold_label'].tolist()
val_label = val_data['gold_label'].tolist()
test_label = test_data['gold_label'].tolist()
```
- gold_label열을 리스트화 했다.

```python
idx_encode = preprocessing.LabelEncoder()
idx_encode.fit(train_label)

y_train = idx_encode.transform(train_label) # 주어진 고유한 정수로 변환
y_val = idx_encode.transform(val_label) # 고유한 정수로 변환
y_test = idx_encode.transform(test_label) # 고유한 정수로 변환

label_idx = dict(zip(list(idx_encode.classes_), idx_encode.transform(list(idx_encode.classes_))))
idx_label = {value: key for key, value in label_idx.items()}
print(label_idx)
print(idx_label)
```

```python
class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name, num_labels):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.classifier = tf.keras.layers.Dense(num_labels,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='softmax',
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs[1]
        prediction = self.classifier(cls_token)

        return prediction
```
- 클래스를 정의했다.

```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
with strategy.scope():
  model = TFBertForSequenceClassification("klue/bert-base", num_labels=3)
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])
```
- tpu를 작동 시키기 위한 코드를 작성한 위 모델을 정의하고 optimizer와 loss값을 저장한뒤 훈련시켰다.

```python
early_stopping = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.001,
    patience=2)

model.fit(
    X_train, y_train, epochs=5, batch_size=32, validation_data = (X_val, y_val),
    callbacks = [early_stopping]
)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test, batch_size=1024)[1]))
```