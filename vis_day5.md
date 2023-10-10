# Vis Day5
```python
import numpy as np
import pandas as pd
```
 -  plotly : 인터렉티브한 시각화를 할 수 있는 파이썬 라이브러리
    -  주로 json형식의 데이터를 받는다
```python
import plotly.express as px

df = px.data.gapminder().query("country=='Canada'")
fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
fig.show()
```
- plotly line_chart

## 1. plotly 시각화 방법 두가지 (그래프 생성단계)
- 1) express 모듈 이용 : 간단하게 시각화
- 2) graph_objects 모듈 이용 : 세세하게 시각화

```python
import plotly.graph_objects as go
```
- plotly의 graph_objects를 go입력자로 받았다.
```python
fig = go.Figure(
    data = [go.Bar(x=[1,2,3],y=[1,3,2])],
    layout = go.Layout(
            title = go.layout.Title(text = "A Figure Specified By A Graph Objects")
    )
)
fig.show()
```
- go.Figure가 그림을 그리는 함수라고 생각.
- data 값에 들어가는 go._가 그래프의 형식을 정하고 그 뒤에 (x값 = [], y값 = [])로 데이터 값을 지정한다.
- layout은 그래프와는 상관없이 바깥쪽을 지정할때 사용하는 구문이다.
    - ex) 색상, 축의 수치, 제목등
- 마무리로 fig.show()로 그래프 출력
- graph_objects보다는 express가 초보자에게는 더 많이 사용된다. 혼용해서 사용되는 경우도 다분하다.
```python
import plotly.express as px
```
- plotly의 express를 px입력자로 받았다.
```python
fig = px.bar(x=["a","b","c"], y = [1,3,2], title = "A Figure Specified By express")
fig.show()
```
- Graph_objects는 그래프를 세세하게 시각화를 할 때 사용되고, 간단하게 시각화를 할 때는 express 모델을 사용한다.
- 하나만 사용하는것이 아닌 두가지 모두 혼용해서 사용도 가능하다.
```python
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Bar(x=[1,2,3],y=[1,3,2]))
fig.show()
```
- 빈 그래프 생성 후 add_trace함수를 이용해서 빈 그래프에 값을 넣을 수 있다.
```python
import plotly.express as px

# iris 데이터 불러오기
df = px.data.iris()

# express를 활용한 scatter plot 생성
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 title="Using The add_trace() method With A Plotly Express Figure")
fig.add_trace(
    go.Scatter(
        x = [2,4],
        y = [4,8],
        mode = "lines",
        line=go.scatter.Line(color = "gray"),
        showlegend = False)
)
fig.show()

- add_trace를 이용해서 그래프 내의 중간에 선을 그을수도 있다
```python
from plotly.subplots import make_subplots

# subplot 생성
fig = make_subplots(rows=1, cols=2)
fig.add_scatter(y=[4, 2, 3.5], mode="markers",
                marker=dict(size=20, color="LightSeaGreen"),
                name="a", row=1, col=1)
fig.add_bar(y=[2, 1, 3],
            marker=dict(color="MediumPurple"),
            name="b", row=1, col=1)

fig.add_scatter(y=[2, 3.5, 4], mode="markers",
                marker=dict(size=20, color="MediumPurple"),
                name="c", row=1, col=2)
fig.add_bar(y=[1, 3, 2],
            marker=dict(color="LightSeaGreen"),
            name="d", row=1, col=2)
fig.update_traces(marker=dict(color="RoyalBlue"),
                  selector=dict(type="bar"))
# 그래프의 차트 여러개인 경우 한번에 색을 일괄적으로 바꿀 때 사용
```
```python
import plotly.graph_objects as go

fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))

fig.update_layout(title_text="Using update_layout() With Graph Object Figures",title_font_size=30)

fig.show()
```
- Figure로 x와 y의 값을 주어 바 그래프를 생성했다.
- update_layout를 이용해서 타이틀 레이아웃 추가.
```python
import plotly.graph_objects as go
import plotly.express as px

df = px.data.tips()
x = df["total_bill"]
y = df["tip"]

fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers'))

fig.update_xaxes(title_text='Total Bill ($)')
fig.update_yaxes(title_text='Tip ($)')

fig.show()
```
- 내장데이터인 tips를 이용해서 데이터프레임을 만든 뒤, x에 "total_bill", y에 "tip"값을 주어 scatter그래프를 그린 뒤 축 타이틀을 추가했다.

```python
import plotly.express as px

fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])

fig.update_layout(
    width=600,
    height=400,
    margin_l=50,
    margin_r=50,
    margin_b=100,
    margin_t=100,
    paper_bgcolor="LightSteelBlue",
)

fig.show()
```
- 먼저 그래프를 만든 뒤, margin값을 설정했다. 그 후에 백그라운드 컬러를 지정해 margin이 잘 보이게 작성했다.

```python
import pandas as pd
pd.options.plotting.backend = "plotly"
```
- 이 구문을 이용해서 pandas기본 설정인 matplotlib에서 plotly로 변경이 가능하다.