# KNeighborsRegressor
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
```

- 오늘 사용한 모델들이다. 오늘은 추가로 Regressor라는 모델을 배웠는데 K개의 샘플을 통해 값을 예측하는 방식이다.

```python
movie_dataset=pd.read_csv("title.basics.tsv", delimiter="\t")
movie_ratings = pd.read_csv("title.ratings.tsv", delimiter = "\t")
movie_dataset = movie_dataset[:100]
movie_ratings = movie_ratings[:100]
```
- 교수님이 주신 파일들을 다운받은 뒤, 파일이 너무 커 각각 0~99번까지의 데이터만을 다시 재정의해줬다.

```python
movie_dataset=movie_dataset[['startYear','runtimeMinutes']]
movie_dataset['runtimeMinutes'][movie_dataset['runtimeMinutes'] == '\\N'] = 1
movie_dataset = movie_dataset.astype('int')
movie_ratings = movie_ratings.averageRating
```
- 문자화 되어있는 결측값의 데이터값을 1로 변환한 뒤, movie_dataset를 정수로 변환시켰다.
- 그리고 movie_ratings는 averageRating 열로 재정의했다.

```python
model = KNeighborsRegressor(n_neighbors=5, weights = "distance")
model.fit(movie_dataset, movie_ratings)
```
- KNeighborsRegressor로 모델을 만들고 5개의 이웃값을 받은 뒤, 이때 weights 의 값으로 준 distance의 의미는 거리를 고려한 가중치 평균이다.
- 모델을 movie_dataset과 movie_ratings로 학습시켰다.

```python
model.predict([[1892, 2], [2020, 6], [1900, 1]])
```
- 다음과 같은 예측에도 분류에 성공했다.

```python
diabetes = pd.read_csv("diabetes.csv")
diabetes["Age_Bin"] = pd.qcut(diabetes["Age"], 10, labels = False)
diabetes["BMI_Bin"] = pd.qcut(diabetes["BMI"], 15, labels = False)
diabetes["BloodPressure_Bin"] = pd.qcut(diabetes["BloodPressure"], 16, labels = False)
diabetes["Glucose_Bin"] = pd.qcut(diabetes["Glucose"], 15, labels = False)
total = diabetes[["DiabetesPedigreeFunction", "Age_Bin", "BMI_Bin", "BloodPressure_Bin", "Glucose_Bin", "Outcome", "Insulin", "Pregnancies", "SkinThickness"]]
train = total[2:602]
xtrain = train.drop(columns=['Outcome'])
ytrain = train['Outcome']
xtest = total[602:].drop(columns=['Outcome'])
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)
clf = KNeighborsClassifier() 
params ={'n_neighbors':[3,5,7,9,11,13,15,17,19]}
gs=GridSearchCV(clf, param_grid=params , cv= 5, scoring='roc_auc')
gs.fit(xtrain,ytrain) # 그리드 서치 추출/확인
print(gs.best_score_) # 0.81의 정확도 
print(gs.best_estimator_) #17개의 근접데이터가 적합함
pred = gs.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(answer['Outcome'][602:],pred)
```

- 오늘의 과제였다. 당뇨병 환자들에 대한 데이터로 해당 환자가 당뇨병인지를 판단하는 모델을 만드는 과제였다.
    - 먼저 corr()을 이용해서 상관계수를 확인해 정말 상관이 없는 값들을 제거할까 했으나 내 생각에는 각각의 모든 데이터가 서로에게 영향을 줄것이라 생각해 결국 모든 값을 가지고 모델을 만들었다.
        - 수치값들이 너무 큰 값들은 구간만 나누어주고, Insulin 과 SkinThickness는 0의 값들이 너무 많아서 구간을 나눌수가 없어 그냥 값을 줬다.
        그렇게 만들어진 모델의 정확도가 0.81이었고, 정답률을 0.789가 나왔다.