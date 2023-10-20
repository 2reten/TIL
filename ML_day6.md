# 랜덤포레스트

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from datetime import datetime
```
## 자전거 대여 사용자수 파악하기 (feat.kaggle)
```python
train = pd.read_csv("bike-sharing-demand/train.csv")
test = pd.read_csv("bike-sharing-demand/test.csv")
year = train["datetime"].str.split("-").str[0]
month = train["datetime"].str.split("-").str[1]
day = train["datetime"].str.split("-").str[2].str.split(" ").str[0]
df = pd.read_csv("bike-sharing-demand/train.csv", parse_dates = ["datetime"])
train["year"] = train.tempDate.apply(lambda x : x[0].split("-")[0])
train["month"] = train.tempDate.apply(lambda x : x[0].split("-")[1])
train["day"] = train.tempDate.apply(lambda x : x[0].split("-")[2])
```
- kaggle에서 다운한 데이터로 자전거 대여의 데이터를 불러왔다.
    - 거기서 datetime이  object타입이어서 split을 이용해서 각각 년,월,일로 분리해 새로 저장했다.
        - 아래 코드는 다른 방법으로 추출해 데이터프레임에 새로운 열로 저장했다. parse_dates를 이용하면 지정하는 열을 datetime의 타입으로 변경이 가능하다.

```python
import locale
locale.setlocale(locale.LC_ALL, "english")
list(calendar.day_name)
```
- 지역을 설정하는 모듈과 설정하는 함수이다.
    - 먼저 import한 calendar를 이용하면 무슨요일인지도 출력이 가능하다.

```python
s.dt.strftime("%Y년 %m월 %d일") # 날짜형식 -> 문자열
# dt.strptime() : 문자열 -> 날짜형식
datetime.strptime("202310201053", "%Y%m%d%H%M")
"""
%Y : 4자리 수 year
%y : 2자리 수 year
%m : 2자리 수 month (1~9월의 경우, 앞에 0을 채운다)
%d : 2자리 수 date (1~9일의 경우, 앞에 0을 채운다)
%H : 2자리 수 시간 (24-hour clock, 0~9시의 경우, 앞에 0을 채운다)
%M : 2자리 수 분 (0~9분의 경우, 앞에 0을 채운다)
%S : 2자리 수 초 (0~9초의 경우, 앞에 0을 채운다)
"""
```
- 오늘 따로 배운 함수다. strptime을 이용하면 문자열을 날짜형식으로, strftime을 이용하면 반대로 날짜형식을 문자열로 변경이 가능하다.
    - 방법은 위를 참조

```python
train.year = pd.to_numeric(train.year)
train.month = pd.to_numeric(train.month)
train.day = pd.to_numeric(train.day)
train["hour"] = train.tempDate.apply(lambda x : x[1].split(":")[0])
train["weekday"] = train.tempDate.apply(lambda x : calendar.day_name[datetime.strptime(x[0], "%Y-%m-%d").weekday()])
train["hour"] = pd.to_numeric(train.hour)
```
- numeric역시 오늘 처음 배우게 된 함수다. 이걸 이용해서 뒤에 참조한 열을 int타입으로 바꿨다.
    - hour열도 마찬가지로 split을 이용해서 시간 부분만 참조해서 hour열의 값으로 저장했다.
    - weekday의 값도 tempDate를 apply를 이용해서 datetime의 strptime을 이용해서 문자열을 날짜형태로 변경 후 그 부분을 calendar를 이용해서 요일을 표시했다.
        - 마찬가지로 hour도 numeric을 이용해서 int형태로 바꿨다.
```python
train.drop("tempDate", axis = 1, inplace = True)
```
- tempDate의 열을 삭제했다.

```python
fig = plt.figure(figsize = [12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.barplot(x = "year", y = "count", data = train.groupby("year")["count"].mean().reset_index())
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.barplot(x = "month", y = "count", data = train.groupby("month")["count"].mean().reset_index())
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.barplot(x = "day", y = "count", data = train.groupby("day")["count"].mean().reset_index())
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.barplot(x = "hour", y = "count", data = train.groupby("hour")["count"].mean().reset_index())
```
- 각각 년,월,일,시간별로 그룹화를 해 x값에 그 값들을 y에는 count를 넣어 시각화했다.

```python
fig = plt.figure(figsize = [12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.barplot(x = "season", y = "count", data = train.groupby("season")["count"].mean().reset_index())
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.barplot(x = "holiday", y = "count", data = train.groupby("holiday")["count"].mean().reset_index())
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.barplot(x = "workingday", y = "count", data = train.groupby("workingday")["count"].mean().reset_index())
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.barplot(x = "weather", y = "count", data = train.groupby("weather")["count"].mean().reset_index())
```
- 위와 마찬가지로 계절,휴일,근무일,날씨별로 그룹화를 해 시각화했다.

```python
def newSeason(month):
    if month in [12, 1, 2]:
        return 4
    elif month in [3, 4, 5]:
        return 1
    elif month in [6, 7, 8]:
        return 2
    elif month in [9, 10, 11]:
        return 3
train["season"] = train.month.apply(newSeason)
```
- 달의 정렬이 다르게 돼 있어서 각 계절에 맞는 값으로 새로 정렬했다

```python
fig = plt.figure(figsize = [12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.barplot(x = "season", y = "count", data = train.groupby("season")["count"].mean().reset_index())
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.barplot(x = "holiday", y = "count", data = train.groupby("holiday")["count"].mean().reset_index())
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.barplot(x = "workingday", y = "count", data = train.groupby("workingday")["count"].mean().reset_index())
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.barplot(x = "weather", y = "count", data = train.groupby("weather")["count"].mean().reset_index())
```
- 변경된 값을 다시 확인하기 위해서 같은 코드를 재출력했다.
```python
#온도와 count
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.distplot(train.temp,bins=range(int(train.temp.min()),int(train.temp.max())+1))
#평균온도와 count
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.distplot(train.atemp,bins=range(int(train.atemp.min()),int(train.atemp.max())+1))

#습도와 count
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.distplot(train.humidity,bins=range(int(train.humidity.min()),int(train.humidity.max())+1))

#바람속도와 count
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.distplot(train.windspeed,bins=range(int(train.windspeed.min()),int(train.windspeed.max())+1))
```
- 각 값의 최저값부터 최대값+1을 범위로 구간을 나눠 시각화했다.

```python
fig=plt.figure(figsize=[20,20])
sns.heatmap(train.corr(), annot=True, square=True)
```
```python
#시간과 계절에 따른 count
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax1 = sns.pointplot(x='hour',y='count',hue='season',data=train.groupby(['season','hour'])['count'].mean().reset_index())

#시간과 휴일 여부에 따른 count
ax2 = fig.add_subplot(2,2,2)
ax2 = sns.pointplot(x='hour',y='count',hue='holiday',data=train.groupby(['holiday','hour'])['count'].mean().reset_index())

#시간과 휴일 여부에 따른 count
ax3 = fig.add_subplot(2,2,3)
ax3 = sns.pointplot(x='hour',y='count',hue='weekday',hue_order=['Sunday','Monday','Tuesday','Wendnesday','Thursday','Friday','Saturday'],data=train.groupby(['weekday','hour'])['count'].mean().reset_index())

#시간과 날씨에 따른 count
ax4 = fig.add_subplot(2,2,4)
ax4 = sns.pointplot(x='hour',y='count',hue='weather',data=train.groupby(['weather','hour'])['count'].mean().reset_index())
```
- 아래 코드는 주석에 달아놓은대로 시각화를 했다. 위의 heatmap은 상관계수를 알아보기 위해 시각화했다.

```python
train[train.weather == 4]
```
- weather에 4인 값이 단 하나가 있길래 확인했다.

```python
fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,1,1)
ax1 = sns.pointplot(x='month',y='count',hue='weather',data=train.groupby(['weather','month'])['count'].mean().reset_index())
ax2 = fig.add_subplot(2,1,2)
ax2 = sns.barplot(x='month',y='count',data=train.groupby('month')['count'].mean().reset_index())
```
- weather과 month를 그룹을 묶고 hue값을 weather로 주어 각 값별로 색을 다르게 주어 시각화했다
    - 아래 그래프는 month로 그룹화를 하고 달별 사용자 수를 시각화 한 것이다.

```python
train["weekday"] = train["weekday"].astype("category")
train["weekday"].cat.categories
train["weekday"].cat.categories = ["5", "1", "6", "0", "4", "2", "3"]
```
- weekday의 데이터타입을 category로 변환했다. 그리고 그값을 각 값에 맞게 숫자로 변환했다.

```python
from sklearn.ensemble import RandomForestRegressor
```
- 랜덤포레스트 리그레서를 import했다.

```python
windspeed_0 = train[train.windspeed == 0]
windspeed_not0 = train[train.windspeed != 0]
windspeed_0_df = windspeed_0.drop(['datetime', 'windspeed', 'casual', 'registered', 'count', 'holiday', 'workingday', "day", 'weekday'], axis = 1)
windspeed_not0_df = windspeed_not0.drop(['datetime', 'windspeed', 'casual', 'registered', 'count', 'holiday', 'workingday', "day", 'weekday'], axis = 1)
windspeed_not0_series = windspeed_not0["windspeed"]
```
- 마무리 과정이다. 풍속이 0인값과 아닌 값을 나누고 그 값들에서 필요없는 데이터를 drop, 그 후 sereis에는 풍속이 0이 아닌 그룹의 풍속값을 줬다.

```python
rf = RandomForestRegressor()
rf.fit(windspeed_not0_df, windspeed_not0_series)
windspeed_0["windspeed"] = rf.predict(windspeed_0_df)
```
- 모델을 만들고 훈련을 시키고 예측값을 0값에 넣었다.

```python
train = pd.concat([windspeed_0,windspeed_not0], axis = 0)
```
- 그후 그 데이터를 합치고 train이라고 명명했다.

```python
train.datetime = pd.to_datetime(train.datetime)
train = train.sort_values(["datetime"])
plt.figure(figsize = [20, 20])
sns.heatmap(train.corr(), annot = True, square = True)
```
- datetime만이 다른 데이터타입이어서 그 값을 datetime의 형식으로 바꿨다.
    - datetime의 값을 기준으로 정렬하고 heatmap을 이용해서 풍속0의 값을 바꾼 상태로 시각화를 했으나 0.01의 수치만이 증가했다.

```python
plt.figure(figsize = [5,5])
sns.distplot(train.windspeed,bins=range(int(train.windspeed.min()),int(train.windspeed.max())+1))
```
- 마찬가지로 풍속을 기준으로 시각화했다.
### 복습 겸 새로운 데이터변환
```python
train=pd.read_csv("bike-sharing-demand/train.csv")
test=pd.read_csv("bike-sharing-demand/test.csv")
```
- 데이터를 불러왔다.

```python
combine = pd.concat([train,test],axis=0)
combine['tempDate'] = combine.datetime.apply(lambda x:x.split())
combine['weekday'] = combine.tempDate.apply(lambda x: calendar.day_name[datetime.strptime(x[0],"%Y-%m-%d").weekday()])
combine['year'] = combine.tempDate.apply(lambda x: x[0].split('-')[0])
combine['month'] = combine.tempDate.apply(lambda x: x[0].split('-')[1])
combine['day'] = combine.tempDate.apply(lambda x: x[0].split('-')[2])
combine['hour'] = combine.tempDate.apply(lambda x: x[1].split(':')[0])
combine['year'] = pd.to_numeric(combine.year)
combine['month'] = pd.to_numeric(combine.month)
combine['day'] = pd.to_numeric(combine.day)
combine['hour'] = pd.to_numeric(combine.hour)
```
- 데이터를 변환하거나 새로운 열을 만들어 그 값으로 주고, 데이터의 유형을 int로 변환했다.

```python
combine.season = combine.month.apply(newSeason)
combine.weekday = combine.weekday.astype('category')
combine.weekday.cat.categories = ['5','1','6','0','4','2','3']
```
- 아까 정의해둔 함수로 계절을 재정의하고 카테고리화 후 숫자로 변환했다.
```python
dataWind0 = combine[combine['windspeed']==0]
dataWindNot0 = combine[combine['windspeed']!=0]
dataWind0_df=dataWind0.drop(['datetime','windspeed','casual', 'registered','count', 'holiday', 'workingday', 'day', 'weekday', "tempDate"], axis=1)
dataWindNot0_df=dataWindNot0.drop(['datetime','windspeed','casual', 'registered','count', 'holiday', 'workingday', 'day', 'weekday', "tempDate"], axis=1)
dataWindNot0_series=dataWindNot0['windspeed']
```
- 풍속 데이터 변환 작업이다. 
```python
rf2 = RandomForestRegressor()
rf2.fit(dataWindNot0_df, dataWindNot0_series)
pred = rf2.predict(dataWind0_df)
dataWind0["windspeed"] = pred
```
- 모델을 만들고 예측 시킨 후 그 값을 windspeed에 넣었다.

```python
combine = pd.concat([dataWindNot0, dataWind0], axis = 0)
combine["season"].astype("category")
pd.get_dummies(combine["season"])
```
- 두 데이터를 합치고 season을 범주형 데이터로 변환하는 작업이다.

```python
"""
데이터 타입 : 수치형, 범주형
수치형 : 연속형(실수), 이산형(정수)
범주형(종류) : 명목형(순서가 없음, 혈액형, 색깔 등), 순서형(1월~12월, 계절, 학점 등)
get_dummies를 하면 범주형으로 나타내겠다는 의미
"""
```
- 관련 해설

```python
cate_cols = ['season', 'weather', 'weekday', 'year', 'month', 'hour']
drop_cols = ['datetime','casual', 'registered', 'count', 'tempDate', 'day']
for col in cate_cols:
    combine[col] = combine[col].astype("category")
train = combine[pd.notnull(combine['count'])].sort_values("datetime")
test = combine[~pd.notnull(combine['count'])].sort_values("datetime")
ylabel = train["count"]
datetimecol = test["datetime"]
train = train.drop(drop_cols, axis = 1)
test = test.drop(drop_cols, axis = 1)
```
- category로 만들 열과 필요하지 않은 열을 구분해 리스트로 만들고, for문을 이용해서 category화 했다.
    - 그 후에 각각 결측값을 제외하고 정렬, 결측값만을 정렬했다.
        - ylabel에는 count열을 담고 train과 test에 그 값들을 drop했다.

```python
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
```
- gridsearch를 사용하기 위한 모듈이다.
```python
rf = RandomForestRegressor()
rf_params = {"n_estimators": [10, 30, 50, 70, 100, 200]}
grid_cf = GridSearchCV(rf, rf_params, scoring = 'neg_mean_squared_log_error', cv = 5)
grid_cf.fit(train, ylabel)
grid_cf.predict(test)
grid_cf.predict(train)
mysubmission = pd.read_csv("bike-sharing-demand/sampleSubmission.csv")
mysubmission["count"] = grid_cf.predict(test)
mysubmission.to_csv("mysubmission.csv", index = False)
```
- 이렇게 랜덤 포레슽트 모듈을 이용해서 데이터를 예측하고, 그 값을 csv로 변환해서 kaggle에 제출했다. 점수는 0.48점이 나왔다. (약 1200등)
```python
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
gb_params={'max_depth':range(3,21,2),'n_estimators':[10,50,100, 200]}
grid_gb=GridSearchCV(gb,gb_params,scoring='neg_mean_squared_log_error',cv=5)
grid_gb.fit(train,ylabel)
preds = grid_gb.predict(test)
mysubmission['count']=preds
mysubmission.to_csv("mysubmission_grdientboosting.csv", index=False)
```
- 앙상블 기법으로도 시도해봤으나 점수는 훨씬 떨어졌다.