```python
# https://tutorials.pytorch.kr/?_gl=1*1f3w7gi*_ga*NTQxMTA4OTQuMTY1NzM0MzMzMA..*_ga_LEHG248408*MTY1ODA2Mzg0NS4zLjEuMTY1ODA2Mzg0OC41Nw
# 개인적으로 파이토치를 공부할 때 사용하면 좋은 사이트다.
# 파이토치는 딥러닝 프레임웍이다.
# 토치에는 넘파이와 굉장히 유사하다. -> 수학과 관련된 함수가 많다.
# 토치에는 자동미분, 경사하강법, 자동 신경망 구성등 많은 함수가 있다.
# 파이토치에서는 2가지의 주요한 특징이 있다.
# 1. 파이토치는 넘파이와 유사하다. GPU상에서 실행 가능한 N-차원 텐서(텐서는 배열이라고 생각할 수 있다.)
# => 1차원 구조로 구성된 값을 벡터라고 하고 2차원으로 구성된 값을 행렬 3차원부터 구성된 값은 모두 텐서라고 한다.
# 4차원 텐서는 3차원 텐서를 연결시킨것이다.
# 2. 신경망을 구성하고 학습하는 과정에서 자동 미분을 하는 특징이 있다.
# 2차원 텐서를 나타낼때는(표기) batch-size, dim순으로 온다.
# => 행이 batch-size, 열이 dim으로 온다.
# 컴퓨터에서 한번에 처리하는 2차원 텐서의 크기를 물어본다면 그것이 (batch-size, dim)이다.
# 비전(이미지,영상)분야에서는 주로 3차원 텐서가 사용된다. => (batch-size, width, height)
# 자연어 처리 분야에서는 3차원 텐서가 (batch-size, length, dimention)이 된다.
# 문장이 이중리스트 구조로 [[문장1], [문장2]...[문장n]]이 있다고 가정을 하면 먼저 문장들을 단어 단위로 구분한다. 이것은 2차원 구조다.
# 3차원 공간에 각각의 단어들이 임베딩 되어 표현된다. => [[[0.3, 0.7, 1.5], [1.7, 2.3, 3.1]...[]]]으로 나온다.
# 1) [["나는 사과를 좋아해"],
# ["나는 바나나를 좋아해"],
# ["나는 사과를 싫어해"],
# ["나는 바나나를 싫어해"]]

# 2) 문장 -> 단어
# [["나는", "사과를", "좋아해"],
# ["나는", "바나나를", "좋아해"],
# ["나는", "사과를", "싫어해"],
# ["나는", "바나나를", "싫어해"]]

# 3) 문장 -> 단어 -> 수치(벡터 공간 차원)
# ex) 3차원 벡터
# "나는" = [0.1, 0.2, 0.9]
# "사과를" = [0.3, 0.5, 0.1]
# "바나나를" = [0.3, 0.5, 0.2]
# "좋아해" = [0.7, 0.6, 0.5]
# "싫어해" = [0.5, 0.6, 0.7]

# 4) 훈련 데이터(4,3,3) (배치 사이즈, 문장 길이, 단어 벡터 차원)
#[[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.7, 0.6, 0.5]]
#[[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.7, 0.6, 0.5]]
#[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.5, 0.6, 0.7]]
#[[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.5, 0.6, 0.7]]]

# 5)
# * batch size = 1
# 1번째 배치 : 텐서의 크기? (1,3,3)
#[[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.7, 0.6, 0.5]]
# 2번째 배치 : 텐서의 크기? (1,3,3)
#[[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.7, 0.6, 0.5]]
# 3번째 배치 : 텐서의 크기? (1,3,3)
#[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.5, 0.6, 0.7]]
# 4번째 배치 : 텐서의 크기? (1,3,3)
#[[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.5, 0.6, 0.7]]]

# *batch size = 2
# 1번째 배치 : 텐서의 크기? (2,3,3)
#[[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.7, 0.6, 0.5]]
#[[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.7, 0.6, 0.5]]
# 2번째 배치 : 텐서의 크기? (2,3,3)
#[[0.1, 0.2, 0.9], [0.3, 0.5, 0.1], [0.5, 0.6, 0.7]]
#[[0.1, 0.2, 0.9], [0.3, 0.5, 0.2], [0.5, 0.6, 0.7]]]

# 넘파이는 파이썬 기반의 선형대수와 같은 수학적 함수를 제공하는 패키지다.
# 클래스 : 설계 도면과 같은 것, 붕어빵이라면 붕어빵 기계, 벽돌이라면 벽돌공장 등이 해당된다.
# 객체 : 클래스로부터 생성되는 실체
# 설계도면 -> 실제 건물
# 붕어빵 기계 -> 붕어빵
# 벽돌공장 -> 벽돌
# 메서드 : 객체가 수행하는 동작(기능)
# 속성(어트리뷰트) : 객체의 특성
# 객체지향(적) 프로그래밍 :
# ex) 절차지향적 사례 (과거)
# - 붕어빵 기계로 붕어빵을 판매하던 시대가 지나가고 잉어빵 시대가 도래, 붕어빵 기계는 더이상 사용하지 못하게 됨
# -> 잉어빵 기계를 새로 제작
# ex) 객체지향적 사례 (현재)
# - 붕어빵 기계로 붕어빵을 판매하던 시대가 지나가고 잉어빵 시대가 도래,
# 붕어빵 기계를 재사용(붕어빵 기계의 메서드, 속성 중에서 그대로 사용할 것은 사용하고, 변경할 것은 변경, 버릴것은 버린다.)
```
## 넘파이를 이용해서 텐서 만들기와 파이토치를 이용해서 텐서 만들기

### 넘파이를 이용해서 텐서 만들기
```python
import numpy as np
t = np.array([1, 2, 3, 4])
t.ndim 
t.shape
```
- .ndim을 사용하면 텐서가 몇차원인지 알 수 있다.
- .shape는 텐서의 크기를 알 수 있다. 지금은 (4, )로 출력되는데 이는 (1,4)를 의미한다.
- 기본 1차원 배열이 1차원 텐서다.

```python
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
t.ndim
t.shape 
```
- .ndim으로 확인시 2가 출력됐다.
- .shape로 크기 출력값은 (4,3)이 나왔다.
- 마찬가지로 2차원 배열이 2차원 텐서다.

### 파이토치를 이용해서 텐서 만드는 방법

```python
pip install torch
import torch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
```
- pytorch로 텐서를 생성하는 방법이다.

```python
t.dim() # 차원을 출력해준다.
t.shape # 크기를 알려준다. # 속성
t.size() # 크기를 알려준다. # 함수(메서드) # 내차. 달린다() 자동차(클래스)
# '자동차' 클래스(class)로부터 파생(생성)된 실체(객체)가 '내 차'
# 여기서 t는 '내 차'와 같은 실체(객체)가 된다.
# 클래스(FloatTensor): 대문자로 시작
# 즉, FloatTensor이 size를 가지고 있는 것이다.
# class FloatTensor:
#     속성(특성)
#     함수(동작)
```

```python
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
```
- 행렬 연산(덧셈, 뺄셈) : 두 행렬의 크기가 같아야 한다.
- 행렬 연산(곱셈) : 두 행렬의 마지막 차원과 첫번째 차원이 일치하다.

```python
t1 = torch.FloatTensor([[1,1]])
t2 = torch.FloatTensor([[2,2]])
t1 + t2
```
- tensor([[3., 3.]])
    - 행렬간의 연산이다.

```python
t3 = torch.FloatTensor([[1,1]])
t4 = torch.FloatTensor([3])
t3+t4
```
- 자동변경 (브로드캐스팅) : (1, ) => (1,2) 된다.
    - 브로드캐스팅으로 인해 연산이 가능하다.
        - tensor([[4., 4.]])
```python
t5 = torch.FloatTensor([[1,3]])
t6 = torch.FloatTensor([[1],[3]])
t5 + t6
```
- (2,1) + (1,2) 두 배열간의 덧셈이다.
- [1,3] => [[1,3],[1,3]]의 형식으로 변경된다.
    - [[1],    [[1], [1],
    - [3]]     [3], [3]]
        - 내부적으로 브로드캐스팅이 자동으로 변경되니 항상 염두에 두고 연산을 해야한다.
```python
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
#[[1, 2]   .  [[1]      = [[5] (1 * 1 + 2*2)
# [3, 4]]      [2]]        [11]] (3 * 1 + 4 * 2)
```
- 위와 같은 형식으로 연산이 된다.

```python
m1.matmul(m2)
```
- 이도 마찬가지로 같은 값이 출력되는 텐서간의 연산 함수다.

```python
m1*m2
tensor([[1., 2.],
        [6., 8.]])
```
- 위 코드는 자동 브로드캐스팅으로 인해서 m2의 행렬이 변해 결과값이 다르게 출력됐다.

```python
t.mean()
t.mean(dim = 0)
t.mean(dim = 1)
```
- 전체 행렬의 평균이 출력된다.
    - 차원을 인수로 주어 평균을 구할수도 있다.
    - dim = 0은 첫번째 차원, 행렬에서는 행을 의미한다. 행에 해당하는 차원을 제거한다는 의미기도 하다. (2,2) -> (1,2) 열만 남게된다.
    - dim = 1은 첫번째 차원, 행렬에서는 행을 의미한다. 열에 해당하는 차원을 제거한다는 의미기도 하다. (2,2) -> (1,2) 행만 남게 된다.

```python
t= torch.FloatTensor([[1,2],[3,4]])
t.sum()
t.sum(dim=0)
t.sum(dim=1)
t.sum(dim=-1)
```
- sum도 mean과 마찬가지로 차원값을 줘 연산이 가능하다.
    - -1의 의미는 마지막 차원을 뜻한다.
        - 뿐만 아니라 max역시 이와 같은 방식으로 차원을 지정해 출력하는 것이 가능하다.

```python
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)
ft.view([-1, 3])
ft.view([-1, 5])
```
- view함수는 텐서의 크기를 변경하는 함수다.
- 크기를 [-1, 3]으로 변경한다는 의민데 여기서 -1의 의미는 [?, 3]으로 두번째 차원의 값만을 주고 첫번째 차원의 값은 모르니 알아서 실행을 하라는 의미다.
- [2,2,3] 3차원 텐서를 2차원 텐서로 변경(-1, 3)
- 12개의 요소로 구성된 3차원 텐서 -> 열의 개수는 지정해준 값으로 고정시켜 행을 지정하는 방식이다. => (4,3)
    - 마지막 코드는 약수의 관계가 아니라 오류가 발생한다.

```python
ft = torch.FloatTensor([[0],[1],[2]])
ft.squeeze()
ft2 = torch.FloatTensor([[0,1],[1,2],[2,3]])
ft2.squeeze()
```
- squeeze는 크기가 1인 차원을 제거하는 함수다 
    - ```python
         tensor([[0.],
           [1.],
           [2.]])
         tensor([0., 1., 2.])
    ``` 이렇게 변경된다.
    -  ```python
        tensor([[0., 1.],
            [1., 2.],
            [2., 3.]])
        tensor([[0., 1.],
            [1., 2.],
            [2., 3.]])
      ```
    하지만 차원의 값이 1이 없는 텐서는 제거가 되지 않는다.

```python
ft = torch.Tensor([0, 1, 2])
ft.unsqueeze(0)
ft.unsqueeze(1)
```
- unsqueeze는 크기가 1일 차원을 특정 위치에 추가할 때 사용한다.
    - ```python
        tensor([[0., 1., 2.]])
        tensor([[0.],
            [1.],
            [2.]])
    ```

```python
lt = torch.LongTensor([1, 2, 3, 4])
lt = torch.LongTensor([1., 2., 3., 4])
```
- 위의 코드는 정수 아래는 실수로 표현했지만 모두 정수형태로 출력됐다.
    - 실수도 존재하고 정수도 존재하는 경우에는 정수로 통합된다.
```python
lt.float()
bt = torch.ByteTensor([True, False, True, False])
```
- 실수 값을 출력한다.
- True는 1 False는 0으로 출력된다. 추가로 데이터 타입까지도 출력된다.
```python
bt.long()
```
- 데이터타입을 불린이 아닌 정수형으로 변경해 정수타입도 출력이 가능해졌다.

```python
bt2 = torch.LongTensor([12345678901234567, False, True, False])
bt2 = torch.FloatTensor([12345678901234567, False, True, False])
```
- tensor([12345678901234567, 0, 1, 0])
- tensor([1.2346e+16, 0.0000e+00, 1.0000e+00, 0.0000e+00])
    - 같은 의미의 코드다.

```python
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
torch.cat([x,y], dim = 0)
torch.cat([x,y], dim = 1)
```
- cat에서 dim을 0으로 주면 첫번째 차원(상하)을 기준으로 연결된다.
- dim을 1로 주면 두번째 차원(좌우)을 기준으로 연결된다.
    - 딥러닝 시 중간중간 서로 다른 텐서를 연결하는 경우가 있는데 그런 경우에 사용하는 것이다.

## stacking
```python
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])
torch.stack([x,z,y])
```
- 입력한 순서대로 합쳐진다.
```python
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
torch.ones_like(x)
torch.zeros_like(x)
```
- 텐서의 크기는 그대로 두되, 새롭게 0또는 1로 값을 채우는 것이다.

```python
x = torch.FloatTensor([[1, 2], [3, 4]])
x.mul(2)
```
- mul은 그냥 2를 곱한 값을 출력하지만 mul_는 2를 곱한 값을 출력하고 그 값을 저장한다,

```python
result = 0 
def add(num):
  global result
  #result = 1 # 지역변수 : 함수(블록) 내에서만 사용되는 변수
  result += num
  return result
```
- 전역변수를 설정하고 global로 불러와 함수 내부에서 사용했다.

```python
def add1(num):
  global result1
  result1 += num
  return result1
def add2(num):
  global result2
  result2 += num
  return result2
print(add1(3))
print(add1(4))
print(add2(2))
print(add2(6))
```
- 각 값이 (3,7) (2,8)로 출력된다.
    - 이는 값이 누적된다는 의미다.
```python
class Calculator:
    def __init__(self):  # 클래스 -> 객체 생성될 때 자동 호출(생성자)
        self.result = 0
        print("생성자 호출됨")

    def add(self, num):
        self.result += num
        print("add함수 호출됨")
        return self.result
cal1 = Calculator() # 클래스 -> 객체 생성
cal2 = Calculator()
```
- 간단한 더하기 연산이 가능한 계산기 클래스를 만들었다.

## Torch에서 선형회귀 모델 만들기
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
```
- 마지막 manual_seed는 python의 random_seed와 같은 것이다.

```python
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
W = torch.zeros(1, requires_grad = True) 
b = torch.zeros(1, requires_grad = True)
```
- 가중치와 바이어스의 초기값을 설정해 w,b로 저장해줬다.
    - equires_grad = True는 W는 변수라는 것을 의미한다.
```python
hx = W * x_train + b
cost = torch.mean((hx - y_train)**2)
```
- hx와 cost의 계산법을 저장해줬다.

```python
optimizer = optim.SGD([W, b],  lr = 0.01)
optimizer.zero_grad() 
cost.backward()  
optimizer.step()
``` 
- 미분을 해서 구한 기울기를 0으로 초기화해주는 코드다.
- W, b에 대한 기울기가 계산된다.
- W, b에 대한 업데이트된다.

```python
optimizer = optim.SGD([W, b],  lr = 0.01)
for epoch in range(2000):
  hx = W * x_train + b
  cost = torch.mean((hx - y_train)**2)
  optimizer.zero_grad() # 반드시 사용해줘야함
  cost.backward()
  optimizer.step()
  if epoch % 100 == 0:
    print(epoch, W.item(), b.item(), cost.item())
```
- for문을 이용해서 torch로 선형회귀모델을 만들었다.

```python
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```
- 각각 가중치와 bias를 변수로 저장해줬다.

```python
optimizer = optim.SGD([w1, w2, w3, b],  lr = 1e-5) 
for epoch in range(2000):
  hx = w1 * x1_train + w2 * x2_train + w3 * x3_train + b
  cost = torch.mean((hx - y_train)**2)
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()
  if epoch % 10 == 0:
    print(epoch, w1.item(),w2.item(),w3.item(), b.item(), cost.item())
```
-  lr은 보편적으로 0.001 ~ 0.01값을 준다 1e-5 의 의미는 1의 10의 -5승이다.
    - 1 * 10의 -5 승 = 1*  0.00001
-  optimizer.zero_grad()는 반드시 사용해줘야한다.

```pyython
w1.item() * 90 + w2.item() * 90 + w3.item() * 90 + b.item()
```
- 예상 수능점수 : 180.95720041915774로 측정됐다.

```python
# 발산하는 경우
# lr가 클때
# 입력데이터간의 편차가 심한 경우 -> 정규화 혹은 표준화를 해주지 않아서
```
## transpose의 형태로 데이터가 구성되어 있는 경우에 모델 설계

```python
x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  80],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
W = torch.zeros((3,1), requires_grad = True)
b = torch.zeros(1, requires_grad=True)
```
- w1 = torch.zeros(1, requires_grad=True) 이렇게 하나씩 저장하는 것이 아닌 3,1로 행렬값을 주어 한번에 저장도 가능하다.
    - 일반적으로는 하나씩 정의하지 않고 위와 같이 정의한다.
```python
optimizer = optim.SGD([W,b], lr = 1e-5)
for epoch in range(2000):
  hx =x_train.matmul(W) + b
  cost = torch.mean((hx - y_train)**2)
  optimizer.zero_grad() # 반드시 사용해줘야함
  cost.backward()
  optimizer.step()
  if epoch % 100 == 0:
    print(epoch, cost.item())
new_var = torch.FloatTensor([[90, 90, 90]])
model(new_var)
list(model.parameters())
```
- 마무리로 다중선형회귀 모델을 만들어 수능점수를 예측해봤다.
    - ```python
         [Parameter containing:
        tensor([[0.8540, 0.8475, 0.3096]], requires_grad=True),
        Parameter containing:
        tensor([0.3568], requires_grad=True)]
    ```
