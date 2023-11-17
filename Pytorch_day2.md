```python
# PyTorch에서는 대부분의 모델을 만들때 클래스로 만든다.
# model - nn.Linear(1,1)

# 객체지향적인 프로그래밍 언어에서 가장 중요한 포인트 - 상속
# 부모가 자식에게 물려준다고 해서 부모클래스와 자식클래스간의 관계에는 상속관계가 있다.
# 붕어빵 기계(클래스)의 시대가 가고 잉어빵 기계(클래스)를 바라는 사람이 많아졌다.
# 이때 사용하는것이 상속이다. 붕어빵 기계를 부모 클래스라고 부르고 잉어빵 기계를 자식 클래스라고 부른다.
# 부모 클래스 내부에 있는 메서드, 속성, 내부 클래스등을 물려 받는것을 상속이라고 한다.
# 붕어빵 기계에서 중요한 점 : 너비, 높이, 내용물, 굽는 시간 등...과 같은 특징, 메서드 등이 있다.
# 너비를 10, 높이를 5, 내용물은 단팥, 시간은 1분등으로 설정이 되어있다.
# 여기서 이와 같은 내용들을 약간의 수정만을 통해 새로이 잉어빵 기계를 만드는것을 상속이라고 한다.
# 여기서 얻을 수 있는 이점은 시간과 비용의 절감
# agile기법 : 허접하더라도 빠르게 만듬 -> 배포(time to market) 시장 선점 후 기능 개선
# SW는 요구사항 분석, 설계, 구현, 테스트, 배포, 유지보수의 수순을 가진다.(사실 대부분의 개발은 이러한 과정을 가진다.)
# 추상화 : 확장 설계가 가능하도록 설계를 하는것이다. 객체지향 설계에서 굉장히 중요한 부분 중 하나다.
# 최상위 부모 클래스를 루트 클래스라고도 한다.
```

## 클래스로 모델을 구현해보기

```python
import torch.nn as nn
import torch
import torch.nn.functional as F
class LinearRegressionModel(nn.Module):
  def __init__(self) : 
    super().__init__() 
  
    self.linear=nn.Linear(1,1) 
  def forward(self, x):
    return self.linear(x)
```
- 이 코드는 부모 클래스가 nn.Module이다.
    - init는 이 코드에서 LinearRegressionModel클래스의 객체가 생성 되어지는 시점에 자동으로 호출되는 함수다.
    - super는 부모를 지칭하는 것이다. 이 코드의 의미는 부모 클래스 객체를 만드는 것이다.
        - 부모 클래스의 속성이 초기화 된다.
    - input dim이 1차원 output dim도 1차원이라는 의미다.

```python
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
model = LinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
```
- 객체를 생성하는 코드다.
    - optim.SGD는 경사하강법 알고리즘을 적용하는 최적화 함수다.

```python
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

   
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```
- 파이토치에서 제공하는 평균 제곱 오차 함수다.

```python
# 미니 배치 : 대규모 데이터를 작은 단위로 나누어 연산하는 것이다.
# ex)
# 데이터 100건
# 1epoch : 100건의 데이터를 모두 학습
# 20 batch_size ; 한 번에 20건의 데이터를 읽어서 학습
# 5batch(mini) : 20 배치사이즈로 데이터를 5번 가져와서 학습 -> 100건 데이터 학습 -> 전체 데이터를 batch_size로 나눈 값이 batch가 된다.
# 이터레이션 : 1epoch에서 발생하는 w, b의 업데이트 횟수 -> 전체 데이터를 batch로 나눈 값이 이터레이션이다.
# 배치 학습 : 전체 데이터에 대해 한 번에 배치와 함께 업데이트가 된다,
# 미니배치 학습 : 전체 데이터에 대해 여러 번 배치 및 업데이트 수행

# PyTorch에서는 데이터셋(DataSet), 데이터로더(DataLoader)를 제공한다.
# -> 이를 이용해서 데이터를 조금 더 쉽게 데이터 셔플, 미니배치 학습, 병렬 처리등의 연산을 빠르게 할 수 있다.
```
```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
```
- 다중 선형 회귀 모델을 만드는 과정이다.
    - 다중 선형 회귀이므로 input_dim=3, output_dim=1로 지정했다.

```python
model = MultivariateLinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
```
```python
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```
- 아까와 같은 코드다.

## 데이터 로드

```python
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader 
x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
```
- 텐서데이터셋을 불러오는 코드와 데이터로더를 불러오는 코드다.

```python
dataset = TensorDataset(x_train, y_train)
```
- TensorDataset :  텐서 데이터를 전달받아 데이터셋(Dataset) 형태로 변환한다.
- Dataset = 병렬처리나 셔플등을 더 쉽게 해줄 수 있는 형태
- 시계열 데이터가 아니라면 배치를 할 때마다 셔플을 해주는게 정확성이 높아진다.

```python
dataloader = DataLoader(dataset, 2, True)
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
```
```python
nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    x_train, y_train = samples
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
```
- 아까와 크게 다른것이 없는 코드다.

```python
new_var =  torch.FloatTensor([[73, 80, 75]])
pred_y = model(new_var)
print(pred_y)
```
- tensor([[150.8920]], grad_fn=<AddmmBackward0>)

```python
# class CustomDataset(torch.utils.data.Dataset):
#   def __init__(self):
#     # 데이터 전처리
#   def __len__(self):
#     # 데이터 개수
#   def __getitem__(self, idx):
#     # 데이터 1개를 리턴
```

```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class CustomDataset(Dataset): 
  def __init__(self):
    self.x_data = [[73, 80, 75],
                   [93, 88, 93],
                   [89, 91, 90],
                   [96, 98, 100],
                   [73, 66, 70]]
    self.y_data = [[152], [185], [180], [196], [142]]

  def __len__(self):
    return len(self.x_data)


  def __getitem__(self, idx):
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y
```
- Dataset은 abstract클래스다.
- abstract 클래스란? 추상 메서드는 실체가 없는 함수다.
- 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴한다.

```python
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
nb_epochs = 20
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    x_train, y_train = samples
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, batch_idx+1, len(dataloader),
        cost.item()
        ))
```
- 데이터로드를 이용해서 다시 코드를 실습했다.
    - 반복문은 계속 같은 방식으로 돌아간다.
