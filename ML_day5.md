# 의사결정나무
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
- 오늘 사용한 모듈들이다.

```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
```
- 오늘도 역시 타이타닉 데이터로 실습을 했다.

```python
train['Survived'].value_counts(normalize=True)
train['Survived'].groupby(train['Pclass']).mean()
sns.countplot(x=train['Pclass'], hue=train['Survived'])
```
- Survived열의 값의 평균을 구했다 생존율을 파악할 수 있었다.
    - 응용을 해 Pclass의 각 선실의 등급별 생존율을 파악했다.
- x값을 pclass 그래프의 분리를 survived로 해 사망자와 생존자를 구별해 시각화했다.

```python
train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
train['Name_Len'] = train['Name'].apply(lambda x: len(x))
```
- 이름열의 이름만을 name_title에 저장하고, name_len에는 이름의 길이를 저장했다.

```python
train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean()
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
```
- survived를 그룹화하고 각 이름의 길이를 5개의 구간으로 나눠 연령대별 생존자의 비율을 파악했다.
    - ticket_len이라는 열을 ticket의 길이 값으로 정의했다.

```python
train['Ticket_Lett'] = train['Ticket'].apply(lambda x:x[0])
pd.crosstab(pd.qcut(train['Fare'], 5), columns=train['Pclass'])
```
- ticket_lett은 ticket의 첫글자만을 담아서 저장했다.
    - pandas의 crosstab을 이용해서 Fare를 5개의 구간으로 나누고 columns를 train의 pclass로 주어 교차검증표를 만들었다.

```python
train['Cabin_Letter'] = train['Cabin'].apply(lambda x: str(x)[0])
train['Cabin_num'] =train['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
```
- cabin_letter는 cabin의 첫글자만을, cabin_num에는 공백을 기준으로 나눠진 구간의 마지막 구간에서 1번 인덱스부터 마지막 인덱스까지의 값을 저장했다.

```python
train['Cabin_num'].replace('an', np.NaN, inplace = True)
```
- an값을 결측값으로 변환한 뒤 저장했다.

```python
train['Cabin_num'] = train['Cabin_num'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
```
- NaN 값인 경우 그대로 유지, 빈 문자열인 경우 NaN 값으로 설정, 숫자로 된 값인 경우 정수로 변환했다.

```python
def names(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test
train,test=names(train,test)
```
- name_len과 name_title을 정의하는 함수를 만들었다 위에서 사용한 코드와 같으나 다른점으로는 i값의 name열을 삭제하고 그 값을 리턴하는 코드이다.

```python
def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test
train, test = age_impute(train, test)
```
- 이 코드의 목적은 age 열의 누락된 값을 name_title과 pclass의 그룹 평균 값으로 채우고, 누락된 값 여부를 나타내는 Age_Null_Flag 열을 추가하는 것이다.

```python
def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test
train, test = fam_size(train, test)
```
- 가족의 규모를 지정해주고 sibsp와 parch열을 삭제하는 코드이다.
```python
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
```
- 결측값을 평균값으로 채운 뒤 저장했다.

```python
def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test
train, test = ticket_grouped(train, test)
```
- 티켓 정보를 처리하고 각 티켓을 그룹화하여 Ticket_Lett 및 Ticket_Len 열을 추가하는 것이다. 특히, Ticket_Lett 열은 티켓을 그룹화하는 데 사용되며, Ticket_Len 열은 티켓의 길이를 저장한다.

```python
def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    return train, test
train, test = cabin_num(train, test)
```
-  Cabin 열의 정보를 처리하고 Cabin_num 및 Cabin_num1 열을 추가하고 더미 변수로 변환하는 것이다. Cabin_num은 구간별로 분류된 열이며, Cabin_num1은 초기 데이터를 처리한 열이다.

```python
def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test
def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test
def drop(train, test, bye = ['PassengerId']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test
```
- 비슷한 작업이다. cabin의 첫 글자 값 만을 저장해 정의하고 cabin 열 삭제, embarked의 결측값을 s로 채우고,  'train' 및 'test' 데이터프레임에서 'bye' 리스트에 지정된 열(기본적으로 'PassengerId')을 삭제하는 것이다. 함수를 호출하면 해당 열이 삭제된 업데이트된 데이터프레임이 반환된다.

```python
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test
train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked','Ticket_Lett',                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])
```
-  범주형 열에 대한 더미 변수를 생성하고 해당 열을 삭제하여 머신러닝 모델에 적용 가능한 형식으로 데이터를 준비하는 것이다. 함수를 호출하면 수정된 데이터프레임이 반환된다.
    - 이 과정까지가 전처리다.
# 랜덤 포레스트
```python
from sklearn.ensemble import RandomForestClassifier
```
- 랜덤 포레스트의 모델을 만들기 위해 모듈을 import했다.

```python
rf=RandomForestClassifier(n_estimators=1000, min_samples_split=10, min_samples_leaf= 1,
                      random_state=42)
rf.fit(train.iloc[:,1:] , train.iloc[:,0])
pd.concat((pd.DataFrame(train.iloc[:,1:].columns, columns=['var']),
pd.DataFrame(rf.feature_importances_, columns=['importance'])),axis=1).sort_values(by='importance', ascending=False)[:20]
test=test[train_df]
test1=pd.read_csv("test.csv")
pred=rf.predict(test)
pred=pd.DataFrame(pred,columns=['Survived'])
submission=pd.concat((test1.iloc[:,0], pred), axis=1)
submission.to_csv("mysubmission.csv", index=False)
```

