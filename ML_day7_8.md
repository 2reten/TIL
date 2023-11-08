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

# LGBM, SMOTE, AUC
```python
pip install lightgbm
from lightgbm import LGBMClassifier
lgbm_clf = LGBMClassifier(n_estimators = 1000, num_leaves = 64, boost_from_average = False)
```
- 어제에 이어서 같은 데이터셋으로 LGBM기법을 사용했다.
    - 여기서 boost_from_average는 데이터가 불균형하게 분포되어 있는 경우 false를 입력해준다.

```python
lgbm_clf.fit(X_train, y_train)
pred = lgbm_clf.predict(X_test)
lgbm_clf.predict_proba(X_test)
X_train.describe()
```
- 모델을 훈련시키고 test 데이터로 값을 예측했다.
    - proba를 이용해서 퍼센트로도 나타냈다.
- X_train에 대한 기술통계 출력

```python
q25 = np.percentile(X_train["V1"].values, 25)
q75 = np.percentile(X_train["V1"].values, 75)
iqr = q75 - q25
iqr15 = iqr * 1.5
lowest_val = q25 - iqr15 # 하한 바운더리
highest_val = q75 + iqr15 # 상한 바운더리
X_train["V1"][(X_train["V1"] < lowest_val) | (X_train["V1"] > highest_val)].index
```
- 우수한 모델이 되려면 이상치를 제거하는 방법과 표준화를 하는 방법이 있다.
    - 이것은 이상치를 제거하는 과정이다.

# SMOTE 오버 샘플링
```python
pip install -U imbalanced_learn
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 0)
```
- 모듈을 다운하고 모델까지 생성하는 과정이다.

```python
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
pd.Series(y_train_over).value_counts()
```
- 모델을 훈련시키고 그 값의 value를 파악했더니 값이 같은 수로 나타났다.

```python
lr_clf = LogisticRegression()
lr_clf.fit(X_train_over, y_train_over)
lr_clf.predict(X_test)
```
- LogisticRegression을 이용해서 모델을 훈련시키고 그 x_test의 값을 예측했다.

```python
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
- 추가로 사용한 모듈들이다.

```Python
cancer_data = load_breast_cancer()
X_data = cancer_data.data
y_label = cancer_data.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size = 0.2,
                                                    random_state = 20231024)
knn_clf = KNeighborsClassifier(n_neighbors = 5)
rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 42 )
dt_clf = DecisionTreeClassifier()
ada_clf = AdaBoostClassifier(n_estimators = 100)
```
- 스태킹 과정이다. 이 과정을 통해 각각의 모듈들로 얻은 예측값을 하나의 데이터로 만들어 모델을 훈련시키고 답을 추측하는 방법이다.

```python
knn_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)
ada_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
dt_pred = dt_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)
print(accuracy_score(y_test, knn_pred)) #0.9298245614035088
print(accuracy_score(y_test, rf_pred)) #0.9473684210526315
print(accuracy_score(y_test, dt_pred)) #0.9298245614035088
print(accuracy_score(y_test, ada_pred)) #0.9736842105263158
```
- 각각의 모델들을 생성하고 예측해보고 그 값의 정확도를 맞춰보았다. 가장 높은 정확도를 보인것은 adaboost였다.

```python
pred = np.array([knn_pred, rf_pred, dt_pred, ada_pred])
pred = np.transpose(pred)
lr_final = LogisticRegression()
lr_final.fit(pred, y_test)
final = lr_final.predict(pred)
accuracy_score(final, y_test)
```
- 그 모델들로 나온 값들을 array로 만들고 그 값을 transpose를 이용해서 열과 행을 변환, 그리고 그 값을 로지스틱 회귀 모델에 넣어 훈련을 시키고 값을 예측했다.

```python
"""
# Association Rule
Itemset = 항목 집합 ex) 1itemset = {milk}, 3itemset = {milk, bred, diaper}
Supprt count = itemset의 빈도수
Support = 항목집합의 빈도수를 전체 거래수로 나눈것, 지지도 라고도 불린다.
(x와 y를 모두 포함하는 집합의 수) / (전체 거래수)
Frequent itemset = 빈발 항목 집합. 최소 지지도(minimum support threshold(minsup)) 이상에 
해당하는 항목 집합을 빈발 항목 집합이라고 한다.
ex) mnsup을 2라고 설정했다면 2와 같거나 더 큰 빈도수를 가진 항목집합이 빈발 항목 집합이 된다.
Confidence = 신뢰도라고 불린다.(x와 y를 모두 포함하는 집합의 수) / (x의 값을 모두 가진 집합 수)
x라는 상품을 포함하는 거래의 집합중에 y도 포함하고 있는 거래가 얼마나 빈번히 발생하는 지에 대한 척도
연관 분석이 필요한 이유? 대용량 데이터베이스에서 기존에는 발견할 수 없었던
아이템간의 관계를 발견할 수 있다는 장점이 있기 때문이다. -> 마케팅 전략을 세울 수 있다.
고객들이 방문한 웹 페이지들이 있다면 그 페이지들의 관계를 바탕으로 웹페이지를 추천이 가능하다.
Association Rule을 찾는 과정
일단 minsup를 직접 설정해야 함.(잘 설정해야된다.) -> 이 minsup의 값을 만족하는 규칙 내에서 연관규칙을 찾기 때문이다.
그렇게 만들어진 아이템 집합으로 높은 confidence를 가지는 룰을 생성한다.
3**d - 2**(d+1) +1 = 규칙 수 공식
if d = 6 => 729 - 128 + 1 = 602
Apriori = 전체 생성 가능한 룰들 중에서 생성 가능한 룰의 수를 최소로 하기 위한 방법
지지도를 만족한다면 빈발 항목 집합, 만족하지 못한다면 비빈발 항목 집합이라고 한다.
Leverage = 0에 가깝다면, 두 상품은 독립이고, 0보다 크다면, 향상도가 1보다 큰 경우, 두 상품은 연관있다.
Conviction = conviction이 1이면 서로 관련없다. conviction이 1보다 크다면 x가 주어졌을 경우 
y가 발생 하지 않는 경우가 x를 고려하지 않았을 경우보다 줄어들었다는것을 의미한다.
반대로 말하면 x가 y의 발생여부를 예측하는데 유용한 품목이 되는것이다. 
비슷한 논리로 conviction이 1보다 작으면 x는 y의 발생 여부를 예측하는데 유용하지 않은 품목이 되는 것이다.
순차분석 = 어떤것을 먼저 사고 어떤것을 나중에 사는지를 분석 -> 시간과 순서를 고려
"""
```
- 오후에 배운 이론이다.

```python
dataset=[['사과','치즈','생수'],
['생수','호두','치즈','고등어'],
['수박','사과','생수'],
['생수','호두','치즈','옥수수']]
pip install mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
```
- 사용할 모듈과 데이터 셋이다.

```python
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns = te.columns_)
```
- TransactionEncoder를 만들고 훈련시켰다. 그리고 그 값을 transform을 이용해서 true와 false로 출력한 뒤 그 값을 데이터프레임화 columns는 그 값의 이름들을 사용했다.

```python
freq_itemsets = apriori(df, min_support=0.5, use_colnames=True)
res = association_rules(freq_itemsets, metric = "lift")
res[res["lift"] > 1]
```
- 지지도가 0.5이상인 값만을 남기고 colnames를 이용해서 각 데이터에 이름을 표시했다.
    - association_rules를 이용해서 각 신뢰도, 지지도, 향상도 등을 확인하고, 향상도가 1 이상인 값만을 출력했다.

```python
dataset = [['Milk', 'Onion', 'Nutmeg', 'Eggs', 'Yogurt'],
           ['Onion', 'Nutmeg', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Ice cream', 'Eggs']]
te = TransactionEncoder()
te_ary = te.fit(dataset)
te_ary = te_ary.transform(dataset)
df = pd.DataFrame(te_ary, columns = te.columns_)
freq_itemsets = apriori(df, min_support=0.4, use_colnames=True)
res = association_rules(freq_itemsets, metric = "lift")
res[res["lift"] < 1]
```
- 위에서 했던 과정을 새로운 데이터셋으로 다시 복습하는 과정이다.

- minsup의 값을 0.2로 하면 많은 데이터가 출력돼 0.4로 조정
    - 지지도가 0.4였던 corn은 향상도가 1을 넘지 못해 출력되지 않았다.
- Milk가 들어간 빈발 항목 집합은 모두 향상도가 1에 가까운 수치를 출력했고 향상도의 수치가 높은 목록들은 "Onion","Eggs","Nutmeg","Yorgurt" 4가지 뿐이다.
- "Onion","Eggs","Nutmeg","Yorgurt" 4가지 모두가 높은 양의 상관관계를 가지고 있어 유통에 문제가 생긴다면 서로 대체가 가능할 뿐만 아니라 묶음으로 팔기도 좋아 보인다.
- 음의 상관관계를 가지는 값은 ["Milk", "Eggs"], ["Eggs", "Yorgurt"]가 있다.
- 음의 상관관계와 양의 상관관계가 중복되는 값이 있는데 그렇다면 [ "Onion","Eggs","Nutmeg"] 혹은 ["Onion","Nutmeg","Yorgurt"]의 형식으로 묶어서 판다면 괜찮을 것 같다.
- 달걀은 지지도가 0.8 개별로 팔아도 충분히 잘 팔리니 [ "Onion","Eggs","Nutmeg"] 보다는  [ "Onion","Nutmeg","Yorgurt"]쪽이 판매에 더 강세를 보일 수 있다고 생각한다.