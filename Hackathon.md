## 헤커톤 전력예측

```python
import pandas as pd
import tensorflow as tf
```

```python
data = pd.read_csv("dataset/train.csv")
df = data[(data["memberNo"] == "1인가구") & (data["jobType"] == "맞벌이")]
```
- 먼저 사용할 데이터를 불러오고 1인가구이면서 맞벌이라고 표시된 이상치를 찾아봤다.

```python
Q1 = df['power'].quantile(0.25)
Q3 = df['power'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 경계 계산
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 출력
outliers = df[(df['power'] < lower_bound) | (df['power'] > upper_bound)]
print("이상치:")
print(outliers)
data.loc[data["id"] ==1512200300, "jobType"] = "외벌이"
data.loc[data["id"] ==2395950066, "memberNo"] = "4인 이상"
data.loc[data["id"] ==2395960086, "memberNo"] = "4인 이상"
```
- 그렇게 출력된 3개의 데이터를 각각 수치에 맞게 외벌이와 4인 이상으로 변경해줬다.
    - 2500개중에 3개라 유의미하다고는 할 수 없는 부분이었던 것 같다.

```python
test = pd.read_csv("dataset/test.csv")
df = pd.concat([data, test])
```
- 한번에 전처리를 하기 위해 test데이터를 부른 뒤 concat을 이용해서 데이터를 합쳤다.
```python
df['length'] = df['electricalAppliances'].apply(lambda x: len(x.split(',')))
```
- 가구당 사용하는 전자기기의 수를 따로 열을 만들어 값을 저장했다.
    - 추후에 생각하니 단순히 수를 측정하는 것이 아닌 어떤 전자기기가 많이 사용됐는지를 파악하는 것이 더 좋았을 것 같았다.

```python
def unify_separator(value):
    # ', '로 분리되어 있는 경우는 그대로 반환, 그 외에는 ', '로 변환
    return value if ', ' in value else ', '.join(value.split(','))
def usage_time(value):    
    # 쉼표로 분리하고 각 값을 정수로 변환하여 연산 후 반환
    values = list(map(float, value.split(', ')))
    return 5 * values[0] + 2 * values[1]
```
- replace와 같은 역할을 하는 함수와, 스플릿을 한 뒤, 값을 7일치로 만들어주는 함수를 정의했다.

```python
df['computerUsageTime'] = df['computerUsageTime'].apply(unify_separator)
df['tvUsageTime'] = df['tvUsageTime'].apply(unify_separator)
df['airconWithHeatUsageTime'] = df['airconWithHeatUsageTime'].apply(unify_separator)
df['airconUsageTime'] = df['airconUsageTime'].apply(unify_separator)
df['heaterUsageTime'] = df['heaterUsageTime'].apply(unify_separator)
df['elecStoveUsageTime'] = df['elecStoveUsageTime'].apply(unify_separator)
df['elecBlanketUsageTime'] = df['elecBlanketUsageTime'].apply(unify_separator)
```

```python
df['computerUsageTime'] = df['computerUsageTime'].apply(usage_time)
df['tvUsageTime'] = df['tvUsageTime'].apply(usage_time)
df['airconUsageTime'] = df['airconUsageTime'].apply(usage_time)
df['heaterUsageTime'] = df['heaterUsageTime'].apply(usage_time)
df['elecStoveUsageTime'] = df['elecStoveUsageTime'].apply(usage_time)
df['elecBlanketUsageTime'] = df['elecBlanketUsageTime'].apply(usage_time)
```
```python
df['computerUsageTime'] = df['computerUsageTime'].apply(lambda x: x * 100)
df['tvUsageTime'] = df['tvUsageTime'].apply(lambda x: x * 150)
df['airconUsageTime'] = df['airconUsageTime'].apply(lambda x: x * 1100)
df['heaterUsageTime'] = df['heaterUsageTime'].apply(lambda x: x * 600)
df['elecStoveUsageTime'] = df['elecStoveUsageTime'].apply(lambda x: x * 3500)
df['elecBlanketUsageTime'] = df['elecBlanketUsageTime'].apply(lambda x: x * 60)
df['airFryerUsageTimePerWeek'] = df['airFryerUsageTimePerWeek'].apply(lambda x: x * 1400)
```
- 함수를 적용한 뒤 각각 시간당 소비전력을 곱해 1주일당 소비 전력을 파악했다.

```python
df = df.reset_index(drop=True)
df["airconWithHeatUsageTime"].str.split(",")
df["airconWithHeatUsageTime_summer"] = df["airconWithHeatUsageTime"].apply(lambda x : x[:2])
df["airconWithHeatUsageTime_winter"] = df["airconWithHeatUsageTime"].apply(lambda x : x[2:])
```
- 인덱스를 초기화하고 여름과 겨울로 데이터를 구분했다.

```python
df["airconWithHeatUsageTime_summer_0"] = df["airconWithHeatUsageTime_summer"].str[0]
df["airconWithHeatUsageTime_summer_1"] = df["airconWithHeatUsageTime_summer"].str[1]
df["airconWithHeatUsageTime_winter_0"] = df["airconWithHeatUsageTime_winter"].str[0]
df["airconWithHeatUsageTime_winter_1"] = df["airconWithHeatUsageTime_winter"].str[1]
df["airconWithHeatUsageTime_summer_0"] = df["airconWithHeatUsageTime_summer_0"].astype(float)
df["airconWithHeatUsageTime_summer_1"] = df["airconWithHeatUsageTime_summer_1"].astype(float)
df["airconWithHeatUsageTime_winter_0"] = df["airconWithHeatUsageTime_winter_0"].astype(float)
df["airconWithHeatUsageTime_winter_1"] = df["airconWithHeatUsageTime_winter_1"].astype(float)
df["airconWithHeatUsageTime_summer_0"] = df["airconWithHeatUsageTime_summer_0"].apply(lambda x: x *5)
df["airconWithHeatUsageTime_winter_0"] = df["airconWithHeatUsageTime_winter_0"].apply(lambda x: x *5)
df["airconWithHeatUsageTime_summer_1"] = df["airconWithHeatUsageTime_summer_1"].apply(lambda x: x *2)
df["airconWithHeatUsageTime_winter_1"] = df["airconWithHeatUsageTime_winter_1"].apply(lambda x: x *2)
```
- 리스트 구조인 데이터라 각각 한 글자씩 추출해서 타입을 실수형태로 바꾼 뒤 평일에 해당하는 값에는 5를 곱하고 주말에 해당하는 값에는 2를 곱했다.
```python
df["airconWithHeatUsageTime_summer"] = df["airconWithHeatUsageTime_summer_0"]+df["airconWithHeatUsageTime_summer_1"
df["airconWithHeatUsageTime_winter"] = df["airconWithHeatUsageTime_winter_0"]+df["airconWithHeatUsageTime_winter_1"]]
df["airconWithHeatUsageTime_winter"] = df["airconWithHeatUsageTime_winter"].apply(lambda x : x * 1100)
df["airconWithHeatUsageTime_summer"] = df["airconWithHeatUsageTime_summer"].apply(lambda x : x * 1100)
```
- 여름과 겨울을 각각 묶어서 1주일로 통일하고 apply를 이용해서 온냉방기의 소비전력을 곱해줬다.

```python
df = df.drop(["airconWithHeatUsageTime_winter_1", "airconWithHeatUsageTime_winter_0", "airconWithHeatUsageTime_summer_0", "airconWithHeatUsageTime_summer_1"], axis = 1)
```
- 더 이상 필요 없다고 판단된 열은 제거했다.

```python
df['memberNo'] = df['memberNo'].apply(lambda x: 1 if '1인가구' in x else (2 if '2~3인 가구' in x else 3))
city_mapping = {'여수시': 1, '순천시': 2, '나주시': 3, '광양시': 4, '목포시': 5}
df['location'] = df['location'].map(city_mapping)
jobtype_mapping = {'외벌이': 1, '맞벌이': 2, '노령(상주)': 3}
df['jobType'] = df['jobType'].map(jobtype_mapping)
houseType_mapping = {'아파트': 1, '빌라': 2, '단독주택': 3}
df['houseType'] = df['houseType'].map(houseType_mapping)
houseArea_mapping = {'80m²(24평) 이하': 1, '116m²(35평) 이하': 2, '116m²(35평) 초과': 3}
df['houseArea'] = df['houseArea'].map(houseArea_mapping)
covidEffect_mapping = {'아니오': 1, '온라인 학습': 2, '재택근무': 3}
df['covidEffect'] = df['covidEffect'].map(covidEffect_mapping)
```
- 매핑을 이용해서 데이터를 카테고리화의 초석으로 만들었다.
## 카테고리화 

``` python
df = pd.get_dummies(df, columns=['covidEffect'], prefix=['covidEffect'])
df = pd.get_dummies(df, columns=['memberNo'], prefix=['memberNo'])
df = pd.get_dummies(df, columns=['jobType'], prefix=['jobType'])
df = pd.get_dummies(df, columns=['houseArea'], prefix=['houseArea'])
df = pd.get_dummies(df, columns=['location'], prefix=['location'])
df = pd.get_dummies(df, columns=['houseType'], prefix=['houseType'])
```
```python
df = df.drop(labels=['airconWithHeatUsageTime', 'electricalAppliances','month1', 'month2','month3', 'month4', 'month5', 'month6', 'month7', 'month8', 'month9','month10', 'month11', 'month12'],axis=1)
```
- 카테고리화와 사용하지 않는 열 제거하는 코드다.
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df = df.drop(["id"], axis = 1)
train = df[:2500]
test = df[2500:]
answer = train["power"]
test= test.drop(labels = "power", axis = 1)
train_columns = train.columns
test_columns = test.columns
train_scale = scaler.fit_transform(train)
test_scale = scaler.fit_transform(test)
train = pd.DataFrame(train_scale, columns = train_columns)
test = pd.DataFrame(test_scale, columns = test_columns)
train["power"] = answer
```
- minmax scaler를 이용해서 스케일링을 하기 위한 전처리다.
    - 먼저 train,test를 분리하고 정답인 power열은 따로 빼둔 뒤  test에서도 결측치만 있을 뿐이라 제거헀다.
    - columns를 각각 저장하고, scale값을 저장한 뒤 그 값들로 데이터 프레임을 만들고 power열을 answer값으로 채웠다.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
from sklearn.metrics import mean_squared_error
X = train.drop(labels='power',axis=1)
Y = train['power']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=100)
model = LinearRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
preds = [math.ceil(pred) for pred in pred]
model.score(X_test, Y_test)
mean_squared_error(Y_test, pred, squared = False)
test_pred = model.predict(test)
test_preds = [math.ceil(test_pred) for test_pred in test_pred]
id_ = pd.read_csv("dataset/test.csv")
submission = id_['id']
test_preds_frame = pd.Series(test_preds, name='power')
submission = pd.concat([submission, test_preds_frame], axis=1)
submission.to_csv('submission_test.csv', index=False)
```
- 모델을 돌리고 저장하는 코드다.
