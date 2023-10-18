# Kmeans
```python
from sklearn import datasets
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
```

- 오늘 사용한 모듈들이다. 배운 Kmeans 모델을 만들기 위해 KMeans까지 함께 import했다.

```python
iris = datasets.load_iris()
iris.key()
labels = pd.DataFrame(iris.target)
data = pd.DataFrame(iris.data)
labels.columns = ["labels"]
data.columns = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
```
- 먼저 pandas에 있는 iris데이터셋을 iris에 저장한 뒤, key값들을 확인하고 그중 labels를 target 열만을 가져와 저장했다.
- data에는 iris의 데이터값을 넣어 저장하고 각각 columns에 값을 줘서 저장했다.

```python
data = pd.concat([data, labels], axis = 1)
feature = data[["Sepal length", "Sepal width"]]
```
- 두 데이터를 옆으로 이어 하나의 데이터로 만들고 feature 라는 변수에 Sepal length, Sepal width를 주어 저장했다.

```python
model = KMeans(n_clusters=3)
model.fit(feature)#데이터에 대한 훈련
pred = pd.DataFrame(model.predict(feature))
pred.columns = ["predict"]
res = pd.concat([feature, pred], axis = 1)
```
- KMeans모델을 만들고 군집수의 값을 3으로 줬다.
- 따로 만들어둔 feature로 모델을 훈련시키고 그 값을 predict로 예측한 뒤, pred라는 변수에 저장하고 columns값을 predict 로 주었다.
    - 그 후 res라는 변수에 feature과 pred를 합쳐 하나의 데이터프레임으로 만들었다.

```python
centers = pd.DataFrame(model.cluster_centers_,
                       columns = ["Sepal length", "Sepal width"])
c_x = centers["Sepal length"]
c_y = centers["Sepal width"]
plt.scatter(res["Sepal length"],res["Sepal width"], c = res["predict"])
plt.scatter(c_x, c_y, c = "r", marker = "D")
pd.crosstab(data["labels"], res["predict"])
print(123/150) # 0.82
```
- centers라는 변수를 cluster_centers_라는 모델의 함수를 이용해서 그 군집의 랜덤한 중앙값을 추려 그 값을 데이터프레임으로 만들었다.
    - 각각 그 값들을 c_x, c_y로 저장하고 res의 sepal length와 sepal widthfmf 각각 주고 c(color)로 res["predict"]를 주어 각 군집별로 색을 나눴다.
        - 추가로 c_x와 c_y를 붉은색 다이아몬드 모양으로 지정해 scatter에 표시했다.

- crosstab은 교차검증이다. 몇개나 맞았는지를 확인할 수 있었다.
```python
[[0, 50, 0
12, 0, 38
35, 0, 15]]
``` 
 의 값이 나왔고 저중 가장 큰 50, 38, 35가 알맞게 형성된 군집이다.

 # SNS 데이터를 바탕으로 학생 클러스터 생성

```ptyhon
data = pd.read_csv("snsdata.csv")
data.describe(include = "object")
data["gender"].value_counts(dropna = False)
data["gender"].fillna("not disclosed", inplace = True)
```
- 다운해둔 데이터를 가지고 와 그 데이터의 기술통계를 확인했다.
    - 기술통계는 수치형 데이터만을 확인할 수 있었다.

- 성별의 각 값들의 수를 파악하였고 거기서 결측값을 not disclosed로 변환했다.

```python
data["age"].isnull().sum() #5086
data.groupby("gradyear")["age"].mean()
data["age"] = data.groupby("gradyear")["age"].transform(lambda x : x.fillna(x.mean()))
```
- 데이터 상에서 age열의 결측값의 수를 확인했다.
    - 연도별 age의 평균값을 구하고 그 값을 transform을 이용해서 각 년도별 결측값으로 채워넣었다.

```python
sns.boxplot(data["age"])
q1 = data["age"].quantile(0.25) # 16.504
q3 = data["age"].quantile(0.75) # 18.391
iqr = q3 - q1
q1 - 1.5 * iqr # 13.672 
q3 + 1.5 * iqr # 21.222
```
- boxplot 을 이용해서 데이터의  outliers를 먼저 파악했다.
    - quantile을 이용해서 q1과 q3값을 알아내고 q3-q1으로 iqr값을 구했다.

```python
df = data[(data["age"] > (q1 - 1.5 * iqr)) & (data["age"] < (q3 + 1.5 * iqr))] #29633 rows × 40 columns
```
- outliers 를 제외한 정상 범위의 데이터만 추출하여 데이터프레임 구성했다.

```python
names = df.columns[4:]
scaled_feature = data.copy()
features = scaled_feature[names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
```

- 학생정보에 관한 앞쪽의 3번까지의 데이터를 제외하고 나머지의 데이터로 names를 정의했다.
    - scaled_feature을 data를 카피해 정의하고 features는 scaled_feature의 names열만을 가져와 저장했다.
        - 그리고 스케일링 작업을 시작하고 표준화까지 했다.

```python
scaled_feature[names] = features
def gen_to_num(x):
    if x == "M":
        return 1
    if x == "F":
        return 2
    if x == "not disclosed":
        return 3
scaled_feature["gender"] = scaled_feature["gender"].apply(gen_to_num)
kmeans = KMeans(n_clusters = 5, random_state = 42)
model = kmeans.fit(scaled_feature)
model.labels_
model.cluster_centers_
```

- features의 값을 scaled_feature의 names열의 값으로 받고 남자라면 1 여자라면 2 not disclosed값이면 3을 반환하는 함수를 정의했다.
    - 그 후 성별열을 apply함수를 이용해서 값을 변환하고 군집수는 5, random_state는 42를 주어 모델을 훈련시켰다.

```python
wcss = [] # within cluster sum of square
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++' ,max_iter=300,random_state=0)
    kmeans.fit(scaled_feature)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,20),wcss) 
plt.title('THe Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel("WCSS")
plt.show()
```
- 시각화다. 20까지의 군집수를 반복하며 팔꿈치 부분의 값이 적절한 군집의 수라고 배웠다.

```python
kmeans = KMeans(n_clusters = 9, random_state = 41)
kmeans.fit(scaled_feature)
data["cluster"] = kmeans.labels_
plt.figure(figsize=(12,7))
axis = sns.barplot(x=np.arange(0,9,1), y=data.groupby(['cluster']).count()['age'].values)
x=axis.set_xlabel("cluster Number")
x=axis.set_ylabel("Number of students")
data.groupby(['cluster']).mean()[['basketball', 'football','soccer', 'softball','volleyball','swimming','cheerleading','baseball','tennis','sports','cute','sex','sexy','hot','kissed','dance','band','marching','music','rock','god','church','jesus','bible','hair','dress','blonde','mall','shopping','clothes','hollister','abercrombie','die', 'death','drunk','drugs']]
```
- 최적의 군집값은 아니지만 군집의 수를 9로 정하고 random_state값은 41, 그리고 그 모델을 훈련시켰다.
    - 이후 kmeans의 labels_값은 data의 cluster로 정의하고 seaborn으로 x값에 군집의 수, y값을 각 군집의 학생 수로 정의했다.

- 마무리로 cluster을 기준으로 그룹화해 각 열의 평균값을 확인했다.
 -> 평균의 값이 1에 가까울수록 언급을 많이 했다는 의미