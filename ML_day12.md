# 추천 시스템

```python
df1 = pd.read_csv("tmdb_5000_credits.csv")
df2 = pd.read_csv("tmdb_5000_movies.csv")
df1.columns = ['id', 'title', 'cast', 'crew']
df = df2.merge(df1, on = "id")
c = df["vote_average"].mean()
m = df["vote_count"].quantile(0.9)
```
- 파일을 불러오고 df1의 열이름을 변경, 그 후 df2와 합쳐 df로 정의했다.
    - 그 과정에서 vote_average의 평균은 c, vote_count의 90%에 해당하는 수치는 m이라고 지정했다.

```python
q_movies = df.loc[df["vote_count"] >= m]
q_movies.shape
```
- q_movies는 vote_count의 값이 m보다 큰 즉, 90% 이상에 해당하는 값들만 저장했다.
    - shape는 (481,23)

```python
def weighted_rating(x, m=m, c=c):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * c)
q_movies["score"] = q_movies.apply(weighted_rating, axis = 1)
```
- x를 변수로 받아 x의 vote_count는 V, x의 vote_average는 R로 저장했다. 그리고 점수를 계산하는 식을 지정해둬 그 값을 return하는 함수를 만들었다.
    - q_movies["score"]는 q_movies에 모든 행에 대해 미리 만들어둔 함수를 적용시켜 새로 score점수를 지정했다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = "english")
df["overview"] = df["overview"].fillna("")
tfidf_mat = tfidf.fit_transform(df["overview"])
```
- Tfidf를 만들때부터 불용어를 처리하고 만들었다.
    - overview에 결측값은 공백으로 채우고 fit_transform을 이용해서 데이터의 스케일을 조정했다.
    - 조정한 값을 tfidf_mat에 저장했다.

```python
from sklearn.metrics.pairwise import linear_kernel
cos_sim = linear_kernel(tfidf_mat, tfidf_mat)
indices = pd.Series(df.index, index=df["title_x"])
```
- 코사인유사도를 계산하기 위한 함수다.
    - 코사인 유사도를 cos_sim에 넣고 df의 index를 데이터로 인덱스를 제목으로 지정해 series로 만들어 indices로 저장했다.

```python
def get_recommendation(title, cosine_sim = cos_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)
    sim_scores = sim_scores[1:11]
    m_indices = [i[0] for i in sim_scores]
    return df["title_x"].iloc[m_indices]
get_recommendation("The Avengers")
```
- 이 함수는 제목을 입력하면 가장 유사한 영화 10가지를 출력하는 함수다.
    - 아까 indices의 title열을 idx로 받고, 코사인유사도에 idx를 인덱스번호로 지정해 enumerate를 이용해 인덱스 번호와 그 값을 묶어서 리스트화했다.
    - 그 리스트화한 값을 정렬하고 인덱스 1번부터 11번까지의 값을 sim_scores로 저장했다.
        - m_indices는 리스트 컴프리헨션을 이용해서 sim_scores에 들어있는 값 하나하나에 0번 인덱스에 위치하는 값을 저장했다.
        - 그리고 그 값을 df["title_x"]의 인덱스 번호로 출력시켰다.

```python
for f in df[["cast", "crew", "keywords" , "genres"]]:
    df[f] = df[f].apply(eval)
def getDirector(data):
    for i in data:
        if i["job"] == "Director":
            return i["name"]
    return np.nan
df["director"] = df["crew"].apply(getDirector)
```
- eval을 이용해서 저 값들을 따로 리스트화 해 df로 저장했다.
    - 이 함수는 직업이 디렉터인 사람의 이름을 출력하고 없다면 nan값을 출력시키는 함수다.

```python
# 선형대수
# pca? 데이터의 공분산 행렬 -> 고유값 분해 -> 고유벡터의 데이터를 선형 변환
# 고유벡터 : pca의 주성분벡터, 입력 데이터의 분산이 가장 큰 방향
# 고유값 : 고유벡터의 크기, 데이터 분산

# 과정
# 1. 원본 데이터의 공분산 행렬
# 2. 공분산 행렬의 고유값과 고유벡터를 구함
# 3. 고유값이 가장 큰 고유벡터를 추출(축소하고자 하는 차원의 수만큼)
# 4. 고유값이 가장 큰 고유벡터를 이용하여 원본데이터를 변환한다.(차원 축소한다.)
```
```python
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

columns = ['sepal_length','sepal_width','petal_length','petal_width']
iris_df = pd.DataFrame(data = iris.data, columns = columns)
iris_df["target"] = iris.target

iris_df.head()
```
- 붓꽃 데이터를 받아와 열이름 지정, 데이터프레임화(데이터는 iris의 데이터)를 한 뒤 target열을 iris.target으로 만들었다.

```python
markers = ["^", "s", "o"]

# 0:setosa, 1:versicolor, 2:virginica
for i, marker in enumerate(markers):
    x_axis_data = iris_df[iris_df['target']==i]['sepal_length']
    y_axis_data = iris_df[iris_df['target']==i]['sepal_width']
    plt.scatter(x_axis_data, y_axis_data, marker=marker, label=iris.target_names[i])

plt.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
```
- 시각화 작업이다. x축과 y축에 각각 sepal_length,sepal_width를 받아 지정된 값별로 ^,o,s로 표시했다.

```python
from sklearn.preprocessing import StandardScaler

iris_f_scaled = StandardScaler().fit_transform(iris_df.iloc[:,:-1])
```
- pca는 미리 표준화를 해줘야한다.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(iris_f_scaled)
iris_pca = pca.transform(iris_f_scaled)
```
- 모듈을 import하고 n_commponents의 값은 주성분의 수 즉, 차원의 수를 의미한다.
    - 만들어둔 모델을 fit을 이용해 훈련시키고, 변환한 값을 iris_pca로 저장했다.

```python
iris_df_pca = pd.DataFrame(iris_pca, columns = pca_columns)
iris_df_pca['target'] = iris.target
```
- pca로 만든 데이터를 이용해서 데이터프레임을 만들고 iris.target값으로 ["target"]열을 만들었다.

```python
markers = ["^", "s", "o"]

# 0:setosa, 1:versicolor, 2:virginica
for i, marker in enumerate(markers):
    x_axis_data = iris_df_pca[iris_df_pca['target']==i]['pca_component_1']
    y_axis_data = iris_df_pca[iris_df_pca['target']==i]['pca_component_2']
    plt.scatter(x_axis_data, y_axis_data, marker=marker,label=iris.target_names[i])

plt.legend()
plt.xlabel('pca_component_1')
plt.ylabel('pca_component_2')
plt.show()
```
- pca로 만들어진 값으로 시각화했다.
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rcf = RandomForestClassifier(random_state=1017)


scores = cross_val_score(rcf, iris_df.iloc[:,:-1], iris_df.target, scoring = "accuracy", cv = 3)

print(f"원본 데이터 fold별 정확도: {scores}")
print(f"원본 데이터 평균 정확도: {np.mean(scores):.4f}")
```

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rcf = RandomForestClassifier(random_state=1017)


scores = cross_val_score(rcf, iris_df_pca.iloc[:,:-1], iris_df_pca.target, scoring = "accuracy", cv = 3)

print(f"pca 데이터 fold별 정확도: {scores}")
print(f"pca 데이터 평균 정확도: {np.mean(scores):.4f}")
```
- 원본데이터와 pca로 만든 데이터의 정확도를 비교했다.
    - 0.1 정도의 정확도 차이가 있었다. 차원이 축소되는 과정에서 데이터의 소실이 일어나 생긴 일이다.

```python
import surprise
from surprise import Dataset
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=0.25, random_state=0)
```
- surprise를 이용해서 한번에 추천 시스템을 만드는 시도를 했다.

```python
import numpy as np
import pandas as pd
import random

from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import GridSearchCV
np.random.seed(20231030)
```

- 사용할 모듈을 import하고 seed값을 주어 값을 고정시켰다.

```python
ratings_data = pd.read_csv("ratings.csv")
```
- ratitngs_data를 불러와 저장해주었다.
    - book_id의 len값은 10000, user_id의 len값을 53424다.

```python
reader = Reader(rating_scale = (1,5))
data = Dataset.load_from_df(ratings_data[["user_id", "book_id", "rating"]],reader)
raw_ratings = data.raw_ratings
random.shuffle(raw_ratings)
```
- reader를 이용해서 데이터를 사용자id, 아이템id, 평점, 시간 순으로 정렬하는 과정이다. 이 과정에서 시간 데이터는 없어 모두 none값이 나왔다.
    - raw_ratings를 data의 raw_ratings로 저장해주ㅜ고 그 값을 shuffle시켰다.

```python
train_test_split_index = int(0.9 * len(raw_ratings)) # 883580
raw_ratings_train = raw_ratings[:train_test_split_index]
raw_ratings_test = raw_ratings[train_test_split_index:]
```
- 인덱스 번호값을 지정하고 그 값을 인덱스 번호로 삼아 train과 test로 나눴다.

```python
data.raw_ratings = raw_ratings_train
test = data.construct_testset(raw_ratings_test)
```
- data변수에 담긴 트레인 데이터로 훈련
    - test 데이터셋 구성

```python
param_grid = {
    "n_factors": [70, 55],
    "n_epochs": [20, 300], 
    "lr_all": [0.005, 0.025, 0.125],
    "reg_all": [0.08, 0.16, 0.32],  
    "random_state": [0],
}
grid_search = GridSearchCV(
    SVD,
    param_grid,
    measures=["rmse", "mae"],
    cv=3,  
    refit=True,
    n_jobs=-1,
    joblib_verbose=2
)
grid_search.fit(data)
```
- grid_search를 이용해 최적의 값을 찾기 위해 파라미터를 지정해주고, 그 값으로 svd를 실행시켜 data를 훈련시켰다.

```python
best_model = grid_search.best_estimator["rmse"]
best_model.predict(uid=10, iid=1000)
```
- best_model을 지정하고 uid(사용자id)와 iid(아이템id)에 값을 줘 평점을 예측하는 코드다.
```python
testset_predictions = best_model.test(test)
accuracy.rmse(testset_predictions)
accuracy.mae(pred)
```
- testset_predictions은 가장 좋은 모델로 테스트를 해 나온 결과고 accuracy.rmse로 testset_predictions의 평균 공분산의 차?를 출력했다.
    - mae는 실제 점수와 예측점수의 절댓값 차다.
```python
svd.predict(uid="371",iid="210")
# 반드시 문자타입으로 변형해 주어야한다.
```
- 마지막으로 uid와 iid에 값을 줄때는 문자열로 주어야한다.
    - int타입으로 준다면 다른 결과가 나온다.