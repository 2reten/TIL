# 연관 분석 추천 시스템

```python
"""
- SVM(support vactor machine) - 딥러닝 등장 이전에 성능이 가장 뛰어났던 모델중 하나이다.
데이터가 부족한 상황에서는 좋은 모델 중 하나.
데이터 분류를 위하여 마진이 최대가 되는 결정 경계선을 찾아내는 머신러닝 알고리즘이다.
결정 경계선은 다른 이름으로 hyperplane(초평면)이라고도 한다.
SVM의 용어 - 결정 경계선, 서포트 벡터, 마진, 비용, 커널 트릭
SVM은 결정 경계선을 그리는데 데이터 사이의 여백(마진)을 최대로 하는 선을 그린다.
분류 뿐만 아니라 연속형 값 역시 예측이 가능하다.
선형 모델 뿐만 아니라 비선형 모델 역시 결정 경계선으로 사용이 가능하다.
2차원에서는 선으로 나오지만, 3차원에서는 2차원 면으로 경계선을 그린다.
데이터의 벡터 공간을 N차원이라고 할 경우, 결정 경계식은 N-1 차원이다.
서포트 벡터 - 벡터는 2차원 공간 상에 나타난 데이터 포인트를 의미
경정 경계선과 가장 가까이 맞닿은 데이터 포인트를 의미한다.
마진 - 서포트 벡터와 결정 경계 사이의 거리를 의미한다.
SVM의 목표는 마진을 최대로 하는 결정 경계를 찾는데 있다.
서포트 벡터의 수는 N차원 이라고 가정을 했을 때 N + 1개만큼 필요하다.
비용 - 얼마나 많은 데이터 샘플이 다른 클래스에 놓이는 것을 허용하는지 결정
Soft margin - 비용이 낮다의 의미는 마진을 최대한 높이고 학습 에러율을 증가시키는 방향으로 결정 경계선을 만든다.
Hard margin - 비용이 높다는 의미는 마진을 최저로 낮추고 학습 에러율을 감소시키는 방향으로 결정 경계선을 만든다.
비용이 너무 낮다면 과소적합, 반대로 너무 높다면 과대적합의 위험성이 존재한다.
커널 트릭 - 1차원 결정 경계는 0차원으로 나타나니 점 하나로 집단을 구분하는 방법은 완벽히 존재하지 않는다.
1차원 데이터를 2차원 공간으로 옮기기
1차원 데이터에 각 값을 제곱을 한 값을 그 값의 y값으로 주고 경계를 분리하고 1차원 결정 경계선응로 분리한다.
N차원 공간에서 구분이 잘 안되는 데이터를 N+1차원으로 옮기고 N차원 평면으로 분리하는 방법이다.
실제로 데이터를 고차원으로 보내지는 않지만 보낸 것과 동일한 효과를 줘 빠른 속도로 결정 경계선을 찾는다.
선형 SVM은 커널을 사용하지 않고 데이터를 분류하고 비용을 조절해서 마진의 크기를 조절할 수 있다.
커널트릭은 선형 분리가 주어진 차원에서 불가능할 경우 고차원으로 데이터를 옮기는 효과를 통해 결정경계를 찾는다.
비용과 gamma를 조절해서 마진을 조절 가능 (RBF SVM)
가우시안 RBF 커널 - 데이터 포인트에 적용되는 가우시안 함수의 표준편차를 조정함으로써 결정 경계선의 곡률 조정
표준 편차 조정 변수를 감마(gamma)라고 부름
감마값이 커지면 경계가 구부러지는 현상이 있음
SVM알고리즘의 장단점
장점 - 특성이 다양한 데이터 분류에 강함. 파라미터를 조정해서 과대/과소적합에 대응 가능
적은 학습 데이터로도 정확도가 높은 분류 성능
단점 - 데이터 전처리 과정이 매우 중요. 특성이 많을 경우, 결정 경계 및 데이터의 시각화가 어려움
"""
```
```python
"""
나이브 베이즈 알고리즘(현재 잘 사용되지 않음)
20세기 초에 많이 사용된 알고리즘
의사 결정 트리와 비슷하다
나이브 베이즈 - 데이터를 나이브(단순)하게 독립적인 사건으로 가정하고,
독립 사건들을 베이즈 이론에 대입시켜 가장 높은 확률의 레이블로 분류를 실행하는 알고리즘
선형회귀 알고리즘
관찰된 데이터들을 기반으로 하나의 함수를 구해서 관찰되지 않은 데이터의 값을 예측하는 것
선형회귀 모델 - 회귀 계수를 선형적으로 결합할 수 있는 모델을 의미한다.
"""
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
- 오늘 사용한 모듈들이다.

```python
data=pd.read_csv("archive/movies_metadata.csv")
data = data.head(5000)
data = data[['id','genres', 'vote_average', 'vote_count','popularity','title', 'overview']]
```
- data의 5000개만 출력해 그 데이터를 다시 data로 저장했다.
    - data[['id','genres', 'vote_average', 'vote_count','popularity','title', 'overview']]로 7개의 열만을 저장했다.
```python
m = data["vote_count"].quantile(0.9) #568
data = data[data.vote_count >= tmp]
C = data["vote_average"].mean()
```
- m을 투표수의 90%의 값으로 지정하고, 불린참조를 이용해서 10%의 데이터만을 저장했다.
    - c를 평균으로 넣었다. 
```python
def weighted_rating(x, m=m, C=C): # weight_rating(x, 568, 7)
    v = x['vote_count']
    R = x['vote_average']
    
    return ( v / (v+m) * R ) + (m / (m + v) * C)
data['score'] = data.apply(weighted_rating, axis = 1)
```
- 이 코드는 한 영화사의 영화의 점수를 매기는 공식이다.

```python
eval(data["genres"][0])[0]["name"]
eval(data["genres"][0])[1]["name"]
eval(data["genres"][0])[2]["name"]
```
- 문자열 형식이지만 보이기에는 이중 리스트의 구조로 되어있어서 그 값을 먼저 eval을 이용해서 리스트화하고 인덱스 번호를 이용해서 출력하는 방식이다.

```python
data['genres'].apply(lambda x: ' '.join([i.get('name') for i in x]))
```
- genres의 모든 데이터를 장르 데이터만을 출력하는 코드다.

```python
tfidf = TfidfVectorizer()
tfidf_mat = tfidf.fit_transform(data["genres"])
tfidf_mat # sparse matrix: 희소행렬 = 요소값이 대부분 0인 행렬 <-> dense matrix
```
- Tfidf는 추천 시스템에서 많이 사용하는 모델이다. 오늘 처음 배운 모듈이라 많이 어려웠다.
    - tfidf_mat은 모델을 장르열만을 fit_transform로 적용해 출력했다.
    - sparse matrix: 희소행렬 = 요소값이 대부분 0인 행렬 <-> dense matrix
```python
tfidf_df = pd.DataFrame(tfidf_mat.toarray(), columns = tfidf.get_feature_names_out())
cosine_similarity(tfidf_df)[0] 
```
- get_feature_names_out을 이용해 tfidf의 모든 unique값만을 추출해 데이터프레임화 시켰다.
    - 그리고 코사인 유사도를 확인했다.
# 여기부터는 과제
```python
score = cosine_similarity(tfidf_df)[0][1:]
score = pd.DataFrame(score)
score.reset_index(drop=True, inplace = True)
title = pd.DataFrame(data["title"][1:], )
title.reset_index(drop=True, inplace = True)
s_list = pd.concat([title, score], axis = 1)
s_list.columns = ["title", "score"]
s_list.sort_values("score", ascending=False).head(10)
```
- 유사도가 높은 10개의 영화의 제목을 출력하는 과정이다.
