# KNN분석

# 데이터 전처리와 분석

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
```
- 처음은 언제나와 같이 사용하는 모듈들을  import한다.
    - 오늘은 sklearn에서 gridsearch와  KNeighborsClassifier도 함께 import했다.
```python
train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")
```
- 금요일과 마찬가지로 train데이터와 test 데이터로 각각 저장해줬다.
```python
trainLen = len(train)
testCopy = test.copy()
```
- 추후에 사용하게 될 변수들을 미리 저장해두었다.

```python
train.info()
total = train.append(test)
```
- info함수로 결측값의 개수를 먼저 확인한 뒤 train과 test를 합쳐 total이라는 변수에 저장했다.

```python
total.isnull().sum()
total[total.Fare.isnull()]
```
- 마찬가지로 결측값의 수를 확인하고 Fare열에 결측값을 확인했다.

```python
total[(total.Pclass == 3)&(total.Embarked == "S")]["Fare"].median()
total["Fare"].fillna(total[(total.Pclass == 3)&(total.Embarked == "S")]["Fare"].median(), inplace = True)
```
- total 변수에서 3등석이고 승선장이 S인 Fare열의 중위수를 구한 뒤 그 값으로 결측값을 채웠다.

```python
total["Title"] = total["Name"].str.extract("([A-Za-z]+)\.", expand = True)
plt.figure(figsize = (8,6))
sns.countplot(x="Title", data = total)
plt.xticks(rotation = 45)
```
- expand를 false로 주면 시리즈형식으로 추출된다 default가 True라는 의미이고 rotation의 값을 주면 지정해준 축이 그 값만큼 틀어져서 나온다.
즉, 겹치는 부분을 겹치지 않게 만들 수 있다.
    - Title을 이름의 호칭부분만 출력해서 저장했다.

```python
mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
total.replace({"Title" : mapping}, inplace = True)
total["Title"].value_counts()
for title in list(total["Title"].unique()):
    print(title)  
```
- mapping에 변수를 넣고  replace를 이용해 Title을 각각 주어진 값에 맞게 저장했다.
    - 그리고 알맞게 저장되었는지 확인 후 unique함수를 이용해서 유일값을 출력했다.

```python
total["Age"].isnull().sum()
total.groupby(["Title"])["Age"].median()
titles = list(total.Title.unique())
for title in titles:
    age = total.groupby("Title")["Age"].median().loc[title] # 각 호칭별 나이의 중위수
    total.loc[(total.Age.isnull()) & (total.Title == title), "Age"] = age
    # 각 호칭별 나이의 중위수로 결측값을 대체
```
- 나이의 결측값의 수를 먼저 파악하고, Title의 unique값을 리스트로 저장해 for문을 이용해서 각 호칭별 나이의 중위수를 age에 다시 저장했다.
    - 전체 데이터에서 age의 결측값이면서 for문으로 하나씩 돌아가는 호칭들이 Title의 값과 같을 때 age를 Age값에 넣었다.

```python
total["Family_Size"] = total["Parch"]+total["SibSp"]
total["Family_Size"].value_counts()
total["Last_Name"] = total["Name"].apply(lambda x: str.split(x, ",")[0])
```
- 먼저 family_size라는 새로운 열을 만들고 그 값을 Parch와 SibSp의 더한 값으로 준 뒤 last_name은 Name열의 성만을 출력해서 저장했다.

```python
dsr = 0.5 # 기본 생존율
total["Family_Survival"] = dsr
len(total.groupby("Last_Name").groups)#875개 그룹
len(total.groupby(["Last_Name", "Fare"]).groups)#982개 그룹
total.groupby(["Last_Name", "Fare"]).groups # 성이 같으면서 운임이 동일한 그룹 생성 => 가족으로 간주
```
- dsr을 기본 생존율로 정의하고 Family_Survival열을 새로 정의하며 모든 값을 dsr로 주었다.
    - last_name을 그룹화로 잡았을 때 875개의 그룹이 나왔고 Fare와 Last_Name을 함께 그룹화했을 때 982개의 그룹이 나왔다.
        - 성이 같고 운임이 동일한 그룹을 가족으로 간주했다.
```python
for grp, grpdf in total.groupby(["Last_Name", "Fare"]):
    print(grp)
    print("-"*50)
    print(grpdf)
    print("="*50)
```
- 오늘 배운 어려운 코드였다. grp와 grpdf를 각 변수로 받고 grp는 Last_Name과 Fare의 값을 받고 grpdf는 그 값에 대한 모든 데이터를 받는다.

```python
for grp, grp_df in total[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId', 'SibSp', 'Parch', 'Age','Cabin']].groupby(['Last_Name', 'Fare']):
    if (len(grp_df) != 1): # 함께 승선한 가족이 있다면
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)["Survived"].max()
            smin = grp_df.drop(ind)["Survived"].min() 
            passID = row["PassengerId"]                        
            if (smax == 1.0): # 나를 제외한 나머지 가족 구성원 중에 생존자가 있다면 "Family_Survival"을 1로 설정
                total.loc[total["PassengerId"] == passID, "Family_Survival"] = 1
            elif (smin == 0.0):
                total.loc[total["PassengerId"] == passID, "Family_Survival"] = 0
```
- grp와 grp_df를 11개의 열을 Last_Name과 Fare열로 그룹화를 하고 grp_df의 len값이 1이상이면 가족과 함께 승선을 했다고 판단했다.
    - iterrows역시 오늘 처음 배운 함수인데 iterrows는 데이터 프레임에만 사용할 수 있는 함수로 데이터 프레임의 각각의 행에 대한 정보를 담는다.
        - 그렇게 되면 ind에는 행인덱스값이 들어가고 row에는 데이터가 들어간다.
    - smax와 smin값을 각각 ind값을 drop한 뒤 최대값과 최소값으로 정의했다. 그리고 만약 나를 제외한 그룹에서 max값이 1이라면 Family_Survival값을 1로 min값이 0이라면 Family_Survival값을 0으로 각각 그 값을 재정의했다.

```python
total.loc[total["Family_Survival"]!= 0.5].shape
```
- 546건, 가족, 친인척등 주변인의 생사여부가 확인된 승객의 수가 546명이다.
```python
pd.qcut(total["Fare"], 5)
total["Fare_Bin"] = pd.qcut(total["Fare"], 5, labels = False)
total["Age_Bin"] = pd.qcut(total["Age"], 4, labels = False)
total.Sex.replace({"male":0, "female":1}, inplace = True)
```
- 수치값이 큰 항목들을 qcut 함수를 이용해 구간을 나누고 그 값을 그 구간의 수로 정의하는 과정이다.
- 성별을 male 이라면 0, female이라면 1로 정의했다.

```python
total = total[['Survived', 'Pclass', 'Sex', 'Age_Bin', 'Family_Size', 'Family_Survival', 'Fare_Bin']]
train = total[:trainLen]
xtrain = train.drop(columns = ["Survived"])
ytrain = train["Survived"].astype(int)
```
- 필요한 열만 출력해 total을 재정의하고, 처음에 만들어뒀던 trainLen으로 슬라이싱을 했다. 그리고 xtrain 은 Survived열을 제거해주었고, ytrain은 Survived 열만 int 타입으로 변환해 저장해주었다.

## 모델링
```python
xtest = total[trainLen:].drop(columns = ["Survived"])
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
```
- xtest를 기존 train의 행들만 가지고 재정의 한 뒤 Survived열을 제거했다.
    - scaler를 standardscaler를 이용해 변수간의 차이가 큰 값들을 평균을 0, 분산을 1로 조정했다.
    - fit을 이용해서 훈련을 시키고, 이후에 한 transform은 완벽한 이해는 하지 못했지만 fit을 기준으로 얻은 mean, variance에 맞춰서 변형하는 함수이다. 실제로 학습시킨 것을 적용하는 메소드

```python
xtest = scaler.transform(xtest)
clf = KNeighborsClassifier()
params = {"n_neighbors": [3,5,7,9,11,13,15,17,19]}
gs = GridSearchCV(clf, param_grid=params, cv = 5, scoring = "roc_auc")
gs.fit(xtrain, ytrain)
print(gs.best_score_)
print(gs.best_estimator_)
pred = gs.predict(xtest)
```
- xtest를 만들어둔 모델에 넣어 값을 출력하고 clf로 KNNeighborsClassifier을 정의했다.
    - params는 gridsearch를 이용하기 위해 값을 정해둔 것이다.
    - gridsearch를 이용할 모델, 값, 주변값의 수(scoring 은 잘 모르겠다.)를 각각 정의했다.
        - gs를 xtrain과 ytrain으로 실행해 최고 정확도와, 파라미터 조정으로 가장 좋은 성능을 보인 모델을 반환한다.
            - 0.8781100992088462
                KNeighborsClassifier(n_neighbors=17)
    - KNN을 훈련시킨 모델에 xtest를 넣어 값을 출력했다.

```python
pd.DataFrame({'PassengerId': testCopy["PassengerId"], "Survived":pred}).to_csv("knn_submission.csv", index = False)
```
- 그 값들을 데이터프레임화 시켰다. passengerid는 이전에 만들어둔  testcopy의 passengerid로 Survived는 pred값으로 만들어 csv파일로 저장해 kaggle에 제출했다.