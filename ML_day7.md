# GradientBoosting

```python
"""
배깅과 부스팅은 모두 트리를 이용하는 방법.
XG Boost : Level-wise tree growth는 균형잡힌 리프 중심의 트리 변화 방식을 사용한다.
수평기반으로 트리를 키워나감 -> 시간이 오래 걸린다.
장점으로는 오버피팅이 많이 일어나지 않는다.
LightGBM : leaf -wise tree growth는 속도가 빠르고 예측 정확도 역시 높은 편에 속한다.
부모가 동일한 노드를 silbling node라고 한다. 또 가장 하단에 있는 노드를 leaf node라고 한다.
로스가 가장 큰 노드를 선택해 서브트리를 구성한다.
배깅은 동일한 방식의 디시젼트리를 만드는 방식이다
병렬로 연결해 학습을 한 뒤 결론을 도출
부스팅은 직렬로 연결해 학습을 한 뒤 결론을 도출
배깅의 대표적인 기법에는 랜덤 포레스트가 있다.
앙상블 기법은 정형적인 데이터들을 분류하는데 적합하다.
RNN, CNN은 비정형 데이터를 분류하는데 많이 사용되는 알고리즘이다.
보팅과 배깅은 여러개의 분류기가 투포를 해서 최종 분류 결과를 도출한다
보팅 : 서로 다른 알고리즘으로 학습된 분류기
배깅 : 같은 알고리즘으로 학습도니 분류기
소프트 보팅 : 모든 분류기들의 레이블 값에 해당되는 확률에 대해 평균 -> 많이 사용됨
하드 보팅 : 다수결 방식(열러개의 분류기가 출력(예측) 결과에 대해 다수결로
부스팅 : 여러개의 분류기가 "순차적"으로 학습 수행 먼저 학습한 분류기가 예측이 틀린 데이터에 대해서
다음 분류기가 올바르게 예측할 수 있도록 가중치를 높게 수정해서 분류하는 방식.
xtrain, ytrain으로 model을 생성하고 x test를 이용해서 predict값을 출력해 ytest와 비교해 정확도를 비교.
스태킹 : data가 있다면 각각 예시(SVM, RF, lightGBM)의 예측값을 출력해 그 값을 train data로 사용한다.
동일한 데이터를 사용하므로 과적합이 일어난다.
CV를 기반으로 하는 stacking : 각각의 모델이 교차 검증 방식으로 학습 데이터를 생성한다.
"""
```
- 오늘 배운 배깅과 부스팅 그리고 보팅과 스태킹이ㅣ다

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
```
- 사용한 모듈들이다. 오늘은 train_test_split이라는 모듈을 처음 사용했다.

```python
data = pd.read_csv("titanic/train.csv")
data['Embarked'].fillna('S', inplace = True)
data['Fare'].fillna(0, inplace=True)
data['Fare'] = data['Fare'].map(lambda x : np.log(x) if x > 0 else 0)
data['Initial'] = data['Name'].str.extract('([A-Za-z]+)\.')
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Other'],inplace=True)
mapping = {
    "Mr":0,
    "Miss":1,
    "Mrs" : 1,
    "Master":2,
    "Other":3
}

data['Initial'] = data['Initial'].map(mapping)
mapping_sex = {
    'male' : 0,
    'female': 1
}

mapping_em = {
    'S' :0,
    'C' :1,
    'Q' :2
}
data['Sex'] = data['Sex'].map(mapping_sex)
data['Embarked'] = data['Embarked'].map(mapping_em)
data.drop(['PassengerId', "Ticket", "Cabin", "Name"], axis = 1, inplace = True)
data.loc[ (data['Age'].isnull()) & (data['Initial'] == 0), 'Age' ] = 32
data.loc[ (data['Age'].isnull()) & (data['Initial'] == 1), 'Age' ] = 28
data.loc[ (data['Age'].isnull()) & (data['Initial'] == 2), 'Age' ] = 5
data.loc[ (data['Age'].isnull()) & (data['Initial'] == 3), 'Age' ] = 45
y = data['Survived']
X = data.drop('Survived', axis = 1)
```
- 지금까지 배웠던 데이터 전처리의 과정과의 차이가 없어 자세한 설명은 생략하겠다. 전과 같은 방식으로 대체하고 결측값을 평균값으로 채우고 x와 y에 각각 정답과 데이터를 넣었다.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("정확도 :{0:.3f}".format(accuracy_score(y_test, pred)))
gb_param_grid = {
    'n_estimators' : [100, 200],
    'max_depth' : [6, 8, 10, 12],
    'min_samples_leaf' : [3, 5, 7, 10],
    'min_samples_split' : [2, 3, 5, 10]
}
gb = GradientBoostingClassifier(random_state=0)
gb.fit(X_train, y_train)
gb_grid = GridSearchCV(gb, param_grid = gb_param_grid, scoring="accuracy", n_jobs= -1, verbose = 1)
gb_grid.fit(X_train, y_train)
```
- 비율을 나타내는 test_size를 0.2의 값을 주어 20%만을 test데이터로 지정했다. train_size를 0.8값을 주어도 같은 결과가 나온다.
    - 랜덤 포레스트 모델을 만들고 X_train과 y_train을 가지고 모델을 훈련시켰다. 그후 predict를 이용해 정답과 비교해 정확도를 출력했다.
    - gradientboosting을 사용하기 위해 파라미터값을 입력하고 그리드서치를 이용해서 최적의 값을 추출한 뒤 그 값으로 모델을 다시 훈련시켰다.

```python
gb_grid.best_score_
gb_grid.best_params_
```
- 각각 최고점은 0.827 최적의 파라미터값은 {'max_depth': 6,
 'min_samples_leaf': 10,
 'min_samples_split': 2,
 'n_estimators': 100} 으로 나왔다.

 ```python
 test = pd.read_csv("titanic/test.csv")
 test['Embarked'].fillna('S', inplace = True)
test['Fare'].fillna(0, inplace=True)
test['Fare'] = test['Fare'].map(lambda x : np.log(x) if x > 0 else 0)
test['Initial'] = test['Name'].str.extract('([A-Za-z]+)\.')
test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr','Other'],inplace=True)
mapping = {
    "Mr":0,
    "Miss":1,
    "Mrs" : 1,
    "Master":2,
    "Other":3
}
test['Initial'] = test['Initial'].map(mapping)
mapping_sex = {
    'male' : 0,
    'female': 1
}
mapping_em = {
    'S' :0,
    'C' :1,
    'Q' :2
}
test['Sex'] = test['Sex'].map(mapping_sex)
test['Embarked'] = test['Embarked'].map(mapping_em)
test.drop(['PassengerId', "Ticket", "Cabin", "Name"], axis = 1, inplace = True)
test.loc[ (test['Age'].isnull()) & (test['Initial'] == 0), 'Age' ] = 32
test.loc[ (test['Age'].isnull()) & (test['Initial'] == 1), 'Age' ] = 28
test.loc[ (test['Age'].isnull()) & (test['Initial'] == 2), 'Age' ] = 5
test.loc[ (test['Age'].isnull()) & (test['Initial'] == 3), 'Age' ] = 45
```
- test데이터로도 실습을 위해 마찬가지의 전처리 과정을 거쳤다.

```python
gb_pred = gb_grid.predict(test)
answer = pd.read_csv("titanic/gender_submission.csv")
answer['Survived'] = gb_pred
answer.to_csv("gbsubmission.csv",index = False)
```
- 이렇게 다시 gradientboosting을 사용해보고 그 결과물을 kaggle에 제출했다. (정답률 : 0.7655)

```python
"""
데이터 불균형 : 클래스가 어느 한 쪽으로만 일방적으로 존재
해결 방법
1) 오버 샘플링 : 클래스가 적은 쪽의 데이터를 랜덤 복원 샘플링하여 복사 붙여넣기를 반복하여
두 클래서의 비율이 비슷하게 조율함
2) 언더 샘플링 : 클래스가 많은 쪽의 데이터를 랜덤 샘플링하여 삭제하기를 반복하여
두 클래스의 비율이 비슷하게 조율함
3) 오버 & 언더 샘플링 : 
ex) Y : 1000건 vs N : 10건 => 1010 / 2 = 505, Y는 505건이 될때까지 언더 샘플링을 수행하고
N은 505건이 될때까지 오버 샘플링을 수행한다.
4) SMOTE 알고리즘 : 기존 데이터를 적절하게 혼합하여 새로운 데이터를 생성하는 방법
랜덤 데이터를 선택 -> KNN 형식으로 K개의 개체 수를 지정
-> 랜덤 데이터와 지정된 K개의 개체와의 사이에 새로운 데이터를 생성
-> 값은 그 연결된 데이터의 값을 따라서 지정된다.
roc curve는 TPR과 FPR 의 비율을 조사하는 그래프이다. 즉, 어떤 모델이 더 좋은지 비교하는 그래프이다
"""
```
- 오후의 수업 과정이었다.

```python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```
- 오후에 사용한 모듈이다.

```python
card_df = pd.read_csv('creditcard.csv')
x_features=card_df.iloc[:,:-1] #x, 284807 rows × 30 columns
y_target=card_df.iloc[:,-1] #y, Length: 284807
xtrain, xtest, ytrain, ytest = train_test_split(x_features, y_target, test_size = 0.3, random_state=20231023, stratify = y_target)
```
- 마지막 데이터의 클래스열만을 제외하고 나머지 값들을 x_features라고 저장하고, 클래스열만을 y_target이라고 저장했다.
    - 비율은 7:3, random_state는 오늘 날자 그리고 stratify는 데이터의 비율을 맞추기 위해 넣어준 값이다.

```python
def get_preprocessed_df(df=None):
    df_copy = df.copy()
    df_copy.drop('Time', axis=1, inplace=True)
    return df_copy
df_copy = get_preprocessed_df(card_df)
```
- Time열만을 빼고 그값을 저장한 뒤 출력하는 함수이다.
    - 그 후 함수를 card_df에 적용시켰다.는

```python
def get_train_test_dataset(df=None):
    df_copy = get_preprocessed_df(df)
    X_features = df_copy.iloc[:, :-1]
    y_target = df_copy.iloc[:, -1]
    X_train, X_test, y_train, y_test = 
    train_test_split(X_features, y_target, test_size=0.3, random_state=20231023, stratify=y_target)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)
```
- 위에서 했던 과정들이다. 클래스열만을 제외한 값을 x-features, 클래스열만을 저장한 값을 y_target으로 저장하고 두 값의 비율을 7:3, random_state값을 오늘 날짜, 그리고 stratify 역시 적용시켰다.

```python
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_clf.predict(X_test)
lr_clf.predict(X_test)
pd.Series(lr_clf.predict(X_test)).value_counts()
lr_clf.predict_proba(X_test)
lr_clf.predict_proba(X_test)[:,1]
```
- 이 코드는 오늘 마지막에 배운 로지스틱 회귀를 사용해본 코드다.
    - 마지막에 predict_proba는 그 값에 대한 확률을 출력하는 코드다.