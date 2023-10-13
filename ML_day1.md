# 머신러닝
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
- 최초로 import한 모듈들이다. 추후에 몇가지 더 import한다.

## 모듈에 사용하기 위한 데이터 다듬기

```python
train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")
```
- kaggle에서 titanic 데이터를 받아 훈련 데이터와 테스트 데이터를 받았다.

```python
train.isnull().sum()
```

- isnull함수를 .sum을 이용해서 결측값의 수를 파악했다.

```python
train["Sex"].value_counts()
sns.countplot(x="Sex", data=train)
```

- 성별을 기준으로 시각화를 했다.

```python
train["Pclass"].value_counts()
sns.countplot(x="Pclass", data = train)
```
- 객실의 등급을 기준으로 시각화를 했다.

```python
train["Embarked"].value_counts()
sns.countplot(x="Embarked", data = train)
```
- value_counts로 수를 파악하고 탑승장을 기준으로 시각화를 했다.

```pythonn
train["Died"] = 1 - train["Survived"]
train["Died"]
```
- train 데이터프레임에 Died라는 열을 추가하고 그 값을 1 - train["Survived"]로 정의했다. 생존자는 1이므로 1 - 1 = 0 , 사망자는 0으로 1 - 0 = 1로 sum을 이용해 사망자만 카운트가 가능해졌다.

```python
train.groupby("Sex").agg("sum")[["Survived", "Died"]].plot(kind = "bar", figsize = (10,5), stacked = True)
```
- 성별로 그룹화를 하고 agg로 sum을 적용시켜 생존자와 사망자를 시각화 하고, 속성값으로 stacked를 주어 그래프를 위로 쌓았다.

```python
plt.hist(train[train["Survived"]==1]["Fare"])
plt.hist(train[train["Survived"]==0]["Fare"])
```
- hist를 이용해 히스토그램으로 시각화를 해봤다.

```python
plt.hist([train[train["Survived"]==1]["Fare"], train[train["Survived"]==0]["Fare"]], stacked = True,
         label = ["Survived", "Dead"], bins = 50)
plt.xlabel("Fare")
plt.ylabel("Number of passengers")
plt.legend()
```
- bins에 50을 주어 구간을 50개를 만들었고 각 라벨에 이름을 주고 레전드값을 주었다.

```python
train2 = pd.read_csv("titanic/train.csv").set_index("PassengerId")
test2 = pd.read_csv("titanic/test.csv").set_index("PassengerId")
df = pd.concat([train2,test2])
```
- 새로  train데이터와 test데이터를 각각 2로 정의해서 합쳐 하나의 데이터프레임으로 만들었다.

```python
pip install missingno
import missingno as msno
```
- 오늘 새롭게 알게된 모듈이다. 이 모듈을 이용해 데이터프레임의 결측치를 확인 할 수 있다.

```python
df["Fare"].fillna(df["Fare"].median(), inplace = True)
```
- 그 후 중위값을 찾는 함수 median을 이용하고 fillna로 결측값을 중위값으로 대체했다.

```python
df["Title"] = df["Name"].str.extract("([A-Za-z]+)\.")
```
- Name열에 각각 A-Z,a-z로 이어진 .이전의 문자만 출력했다.

```python
fig, ax = plt.subplots(1,2, figsize = (18,8))
train["Sex"].value_counts().plot.bar(color = "skyblue", ax = ax[0])
sns.countplot(x = "Sex", hue = "Survived", data = train, ax = ax[1])
ax[0].set_title("num of passengers by sex")
ax[1].set_title("survived vs dead")
```
- 서브플랏을 생성하고, 성별을 기준으로 시각화를 했다.
    - sns에 countplot으로 x값에 성별을 주고 hue값을 기준으로 다른 색상을 준뒤 시각화했다.

```python
pd.crosstab([train["Sex"], train["Survived"]], train["Pclass"],
            margins = True).style.background_gradient(cmap = "summer_r")
```
- crosstab으로 Sex와 Survived를 인덱스로 가져온 뒤 열 인덱스로  Pclass값을 넣어 데이터 프레임을 만들었다.

```python
titles = set()
for name in train["Name"]:
    titles.add(name.split(",")[1].split(".")[0].strip())
```
- 호칭을 정리하는 코드이다. 호칭의 종류만을 남겨 titles에 저장했다.

```python
Title_Dictionary = {"Capt": "Officer","Col": "Officer","Major": "Officer","Jonkheer": "Royalty","Don": "Royalty","Sir" : "Royalty","Dr": "Officer","Rev": "Officer","the Countess":"Royalty","Mme": "Mrs","Mlle": "Miss","Ms": "Mrs","Mr" : "Mr","Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Royalty"}
train["Title"] = train["Name"].map(lambda name : name.split(",")[1].split(".")[0].strip())
train["Title"] = train.Title.map(Title_Dictionary)
```
- Title_Dictionary를 정의하고 Title을 호칭으로만 바꾼 뒤 map함수를 이용해서 리스트 value값으로 변환시켰다.

```python
df1 = train.drop(['Name', 'Ticket', 'Cabin','PassengerId','Died'], axis=1)
```
- Name, Ticket, Cabin, PassengerId, Died는 오늘은 간단히 데이터를 다룬다는 목적으로 지웠다.

```python
df1.Sex=df1.Sex.map({'female':0,'male':1})
```
- 성별을 map을 이용해서 여자는 0, 남자는 1로 값을 변환시켰다.

```python
df1.Embarked=df1.Embarked.map({'S':0, 'C':1, 'Q':2, 'nan':'NaN'})
df1.Title=df1.Title.map({'Mr':0, 'Miss':1, 'Mrs':2,'Master':3,'Officer':4,'Royalty':5})
```
- 호칭과 탑승장 역시 따로 값을 주어 수치화했다.

```python
median_am = df1[df1["Sex"]==1]["Age"].median() #남성
median_aw = df1[df1["Sex"]==0]["Age"].median() #여성
df1.loc[(df1["Sex"]==0)&(df1["Age"].isnull()), "Age"] = median_aw # 여성이면서 나이가 결측값
df1.loc[(df1["Sex"]==1)&(df1["Age"].isnull()), "Age"] = median_am # 남성이면서 나이가 결측값
```
- 각 성별별로 결측값을 중위수로 대체해 값을 채웠다.

```python
df1.dropna(inplace = True)
```
- 결측값이 두개가 존재해 큰 영향이 가지 않는다는 판단하에 그 행을 모두 지웠다.
## 정규화
```python
df1.Age = (df1.Age - df1.Age.min()) / (df1.Age.max() - df1.Age.min())
df1.Fare = (df1.Fare - df1.Fare.min()) / (df1.Fare.max() - df1.Fare.min())
```
- 정규화 작업이다. 공식을 이용해 분모에 최댓값 - 최솟값을 분자에 각값 - 최솟값을 해줬다. 이렇게 해주면 값이 0에서 1 사이로 추려진다. 값이 다른 데이터에 비해 너무 큰 경우 이렇게 정규화를 해준다.
# 모듈 만들기
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit_transform(df1[:])
```
- sklearn으로 분류 알고리즘을 이용해 데이터를 분리하는 모델을 만들었다.

```python
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(df1.drop(["Survived"], axis = 1),
                                                df1.Survived, test_size = 0.2,
                                                random_state = 41,
                                                stratify = df1.Survived)
```
- df1에는 train데이터가 저장, test데이터는 따로 있고, df1의 train데이터를 8:2의 비율로 분리했다.분리된 데이터는 test데이터가 아닌 validation(검증) 데이터이다.
stratify값을 df1.Survived로 주어 Survived의 비율을 8:2로 정의했다. 이것을 층화추출이라고 한다.

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(xtrain, ytrain) # 모델링
```
- 모델링 작업이다. 모델링은 단 두줄만에 정리가 됐으나, 데이터를 가공하는것이 어렵다는것을 깨닳았다. 데이터를 잘 다루고 잘 다듬어야 좋은 모델링을 할 수 있는것이 아닐까 생각했다.

```python
from sklearn.metrics import accuracy_score, confusion_matrix
```
-혼동행렬과 정확성을 알려주는 코드를 import해오는 코드이다.

```python
accuracy_score(ytest, clf.predict(xtest)) # 모델의 정확도 측정
```
- 말 그대로 모델의 정확도를 측정했다. 이 모델로는 약 80%가 나왔다.

```python
confusion_matrix(ytest, clf.predict(xtest))
```
- 혼돈행렬을 알려준다. [[98, 12]
                     [23, 45]]의 값으로 나왔는데 이는 예측값이 실값과 같은 값이 98과 45라는 말이며, 23은 False negative라고 하며 이는 모델이 negative라고 답을 했으나 positive인 경우이고, 12 의 경우는 그와 반대로 모델을 positive라고 답을 했으나 실답이 negative인 경우이다.

- 이후는 가볍게 앞에서 만들었던 test데이터를 지금까지 한 코드들과 같이 새로이 데이터 가공을 해보겠다.

```python
titles = set()
for name in test['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())
test['Title'] = test['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
test['Title'] = test.Title.map(Title_Dictionary)
df2=test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
df2.Sex=df2.Sex.map({'female':0, 'male':1})
df2.Embarked=df2.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'nan'})
df2.Title=df2.Title.map({'Mr':0, 'Miss':1, 'Mrs':2,'Master':3,'Officer':4,'Royalty':5})
df2.Sex=df2.Sex.map({'female':0, 'male':1})
df2.Embarked=df2.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'nan'})
df2.Title=df2.Title.map({'Mr':0, 'Miss':1, 'Mrs':2,'Master':3,'Officer':4,'Royalty':5})
median_age_men2=df2[df2['Sex']==1]['Age'].median()
median_age_women2=df2[df2['Sex']==0]['Age'].median()
df2.loc[(df2.Age.isnull()) & (df2['Sex']==0),'Age']=median_age_women2
df2.loc[(df2.Age.isnull()) & (df2['Sex']==1),'Age']=median_age_men2
df2['Fare']=df2['Fare'].fillna(df2['Fare'].median())
df2.isnull().sum()
df2[df2.Title.isnull()]
df2=df2.fillna(2)
df2.Age = (df2.Age-min(df2.Age))/(max(df2.Age)-min(df2.Age))
df2.Fare = (df2.Fare-min(df2.Fare))/(max(df2.Fare)-min(df2.Fare))
pred = clf.predict(df2)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],
             "Survived" : pred})
submission.to_csv("submission.csv", index = False)
```
- 배우는데 몇시간이나 걸렸던 과정들인데 직접 코드로 적어나가는데는 단 몇분만에 끝나고 모델의 코드가 단 2~3줄만에 끝났다는것이 너무나 충격적이었다. 그래도 데이터가공의 중요성을 다시 한번 깨닫는 중요한 계기가 되었던 하루였다.