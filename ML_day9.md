# 연관분석 기반의 추천 시스템

```python
"""
추천 시스템은 콘텐츠 기반 필터링, 협력 필터링, 하이브리드로 나뉘어진다
그중 협력 필터링은 기억 기반, 모델 기반으로 나눠지며 기억 기반은 최근접 이웃 기반으로도 불린다.
기억 기반은 또 사용자 기반과 아이템 기반으로 나뉜다
잠재요인 기반의 기법은 모델 기반이다.
2천년도 초에는 콘텐츠 기반 필터링이 많이 사용됐다. 그 후에는 협력 필터링이 많이 사용되고 
모델기반(latent factor) - 행렬 분해를 통해서 잠재적인 요인을 추출한다.
콘텐츠 기반 필터링은 사용자가 어떠한 아이템을 선택하면 그 아이템과 가장 유사한 아이템을 추천하는 시스템이다.
-아이템의 속성을 기반으로 추천하기 때문에 이전에 선택 이력이 없는 새로운 아이템도 추천 가능하다
-고객에 대한 데이터가 부족한 경우 추천 성능 보장이 어렵다
협업 필터링(CF)
기억 기반(최근접 이웃 기반) - "특정 아이템에 대하여 선호도가 유사한 고객들은 다른 아이템에서도 비슷한 
선호도를 보일것이다." 라는 전제로 추천을 하는 방식
사용자 기반 협력 필터링 : 사용자 간의 유사도를 측정하여 유사도 높은 이웃이 선택한 아이템 중에서 추천
아이템 기반 협력 필터링 : 아이템 간의 유사도를 측정하여 유사도가 높은 아이템을 추천
모델 기반 - 데이터에 내제 되어있는 복잡한 패턴을 발견하도록 다양한 모델을 활용한 기법.
실제 데이터에 적용했을 때 성능이 우수
"사용자와 아이템 사이에는 사용자의 행동과 평점에 영향을 끼치는 잠재된 특성이 있을 것이다. "
라는 전제로 추천을 하는 방식
크기가 크며 복잡한 데이터로도 쉽고 빠르게 분석 진행이 가능하다.
모델 기반의 성능은 잠재요인의 수에 따라서 성능이 달라진다.
"""
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
chipo = pd.read_csv("chipotle.csv")
```
- tsv = 공백(tab)으로 구분되어져 있는 파일 -> 읽고 싶다면 sep = "\t"를 사용한다.

```python
chipo["quantity"].describe()
chipo["item_price_float"] = chipo["item_price"].str.replace("$", "").astype("float")
chipo["item_name"].unique()
len(chipo["item_name"].unique()) # 50
chipo["item_name"].value_counts() # 메뉴별 주문 횟수
chipo_cost = chipo.groupby(["item_name"])["item_price_float"].mean()
chipo.drop(["order_id"], axis = "columns", inplace = True)
chipo["choice_description"].fillna("Origin", inplace=True)
```
- chipo로 저장한 데이터셋의 quantity열만을 기술통계 함수를 사용해 출력했다.
- replace를 이용해서 $표시를 없애고 float타입으로 변경했다.
    - chipo_cost라는 이름으로 item별 평균가격을 구해서 저장했다.
- order_id는 제거했다.
    - 그리고 choice_description의 추가메뉴를 결측값을 origin값으로 채웠다.

```python
chipo = pd.read_csv("chipotle.csv")
df = chipo[["order_id", "item_name"]]
df["order_id"].max()
df_arr = [[]for i in range(1,df["order_id"].max()+1)] #1~1834+1
```
- 새로 chipo를 만들고 order_id와 item_name만을 df에 저장했다.
    - 고객수만큼의 빈 리스트를 이중구조로 만들었다.
```python
n = 0
for i in df["item_name"]:
    df_arr[df["order_id"][n]-1].append(i)
    n += 1
```
- 먼저 만들어둔 빈 리스트에 각각 인덱스 번호에 맞는 리스트로 값을 넣었다.

```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
```
- 사용할 모듈을 새로 import했다.

```python
te = TransactionEncoder()
te_arr = te.fit(df_arr).transform(df_arr)
te_df = pd.DataFrame(te_arr, columns = te.columns_)
```
- 모델을 훈련시키고 transform으로 True와 False로만 만들어둔 값을 데이터 프레임으로 만들었다.
    - columns로는 원래 이름을 그대로 줬다.

```python
freq_items = apriori(te_df, min_support = 0.02, use_colnames = True)
ar_df = association_rules(freq_items, metric="lift", min_threshold=1)
```
- 어제 했던것과 같은 과정이다. 최소 지지도를 0.02로 지정하고 metric을 lift로 지정해 lift값이 1 이상인 값들만 출력시켰다.
```python
ar_df[ar_df['antecedents'] == frozenset({'Chips'})]
ar_df[ar_df['consequents'] == frozenset({'Chips'})]
```
- 만약 chips가 재고가 떨어지거나 유통에 문제가 생긴다면을 가정하면 만든 코드다.


```python
df = pd.read_csv("ex_data.csv", parse_dates=["Date"])
df["year"] = df.Date.dt.year
df["month"] = df.Date.dt.month
df["day"] = df.Date.dt.day
df["day_of_week"] = df.Date.dt.day_name()
```
- Date열을 datetype으로 만들어서 파일을 불러왔고 그 값에 맞는 년,월,일,요일을 각각 새로 열을 만들어서 저장했다.

```python
fig, (ax,ax2) = plt.subplots(ncols=2)
df["Member_number"].value_counts().head(10).plot(ax = ax, kind="bar", title = "customers who visited the store more often")
ax.set_xlabel("cus id")
ax.set_ylabel("cnt")
df["Member_number"].value_counts(ascending=True).head(10).plot(ax=ax2, kind="bar",
 title = "customers who visited the store less often")
ax.set_xlabel("cus id")
ax.set_ylabel("cnt")
```
- Member_number열을 기준으로 value_counts를 이용해 상위 10개와 하위10개의 barplot을 출력했다.

```python
"""
#1
EDA
가장 많이 방문한 시간 10개 출력
가장 적게 방문한 시간 10개 출력
가장 많이 구매한 상품 10개 출력
가장 적게 구매한 상품 10개 출력
어떤 요일에 구매가 가장 많았는가?
어떤 월에 구매가 가장 많았는가?
"""
# 오늘의 과제
```
```python
df.groupby("Date").nunique().sort_values("Member_number")
df.groupby("Date").nunique().sort_values("Member_number", ascending=True)
df["itemDescription"].value_counts().head(10)
df["itemDescription"].value_counts(ascending=True).head(10)
df["day_of_week"].value_counts() # Monday,Saturday,Friday,Tuesday,Sunday,Wednesday,Thursday
df["month"].value_counts() # 8,5,1,6,3,11,7,10,4,12,2,9
df["year"].value_counts() # 2015
```
```python
pip install apyori
import apyori
```
- 새로 배우게 된 모듈이다.
    - apyori는 연관 규칙 학습 라이브러리로 데이터 집합에서 항목 간의 관계를 찾아내는 기술 중 하나이다.
    - 소비자의 구매 기록과 같은 데이터에서 특정 항목들 간의 연관성을 찾아내어 규칙으로 제시하는 등의 용도로 사용된다.

``` python
df_time = pd.DataFrame(df.groupby("Date")["itemDescription"].nunique().index)
# 일 별 판매된 상품 종류 개수
df_time["prd_num"] = df.groupby("Date")["itemDescription"].nunique().values
df_time["mem_cnt"] = df.groupby("Date")["Member_number"].nunique().values
df_time.set_index("Date", inplace = True)
df_time["items"] = df.groupby("Date")["itemDescription"].unique().values
sns.kdeplot(df_time["mem_cnt"], shade = True)
```
- 각각 일 별 손님 수와 판매된 물건의 수를 정의했다. 그리고 date열을 인덱스로 변경했고, mem_cnt를 기준으로 시각화를 했다.

```python
from apyori import apriori
rules = apriori(transactions=df_time["items"].tolist(), min_support=0.002, min_confidence=0.02, min_lift = 5, max_length=2)
res = list(rules)

pre = [tuple(r[2][0][0])[0] for r in res]
con = [tuple(r[2][0][1])[0]for r in res]

supports = [r[1] for r in res] # support값
confidences = [r[2][1][2] for r in res] # confidences
lifts = [r[2][0][3] for r in res]# lift
res_df = pd.DataFrame(list(zip(pre, con, supports, confidences, lifts)), columns=["pre", "con", "Support", "Confidence","Lift"])
res_df
```
- 코드로는 몇줄이 채 되지않는 짧은 과정이지만 이 과정을 배우는데 굉장히 오래 걸렸다. apriori는 리스트의 형식으로 지지도 신뢰도 향상도 등 모든 값이 다 출력되는데 리스트가 여러개의 구조로 되어있어서 저렇게 인덱스 번호로 찾아서 출력을 하는 과정을 거쳐야한다.
    - 그 과정을 거치고 그 값들을 묶어서 리스트화 한 후 데이터프레임으로 만들어서 출력했다.

- 사실 오늘 연관 분석 기반의 추천 시스템을 알려주신다고 하셨는데 여기까지만 하고 추가로 무언가를 알려주시지 않으셔서 어제와 같은 과정을 복습한 느낌이다.