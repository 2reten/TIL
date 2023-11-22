```python
# GPT를 이용한 텍스트 생성
# 인코더는 가중치를 공유하지 않는다.
# 모든 인코더들중 가장 아래쪽에 있는 인코더는 워드 임베딩이 될 것이다.
# 기본적인 벡터의 크기는 512차원이다. <- 지정 가능하다.
# transformer에서는 인코더에서 자신만의 독립적인 path를 통해서 전달된다.
# transformer에서 selfattention은 각 단어와 다른 단어의 연관된 의미와 같은 understanding요소를 넣어주는 method라고 할 수 있다.
# selfattention은 먼저  queries, keys, values의 3개의 벡터를 만들어 낸다.
# seq2seq의 한계점을 넘기 위해 나온것이 attention이고 그것을 개선시킨것이 self-attention이다.
# GPT2는 특별한 구조를 가지고 있는 것이 아닌 transformer의 디코더부분과 굉장히 유사하다.
# GPT2라는 것은 다음 단어를 예측하는 기능이라고 할 수 있다.
# 디코더 블록은 인코더의 구조를 조금 변경시킨 것이다.
# 파인튜닝은 모델의 가중치를 업데이트해, 수행하고자 하는 특정 태스크의 대한 성능을 더 좋게 만드는 것이다.
```
```python
pip install transformers
import numpy as np
import random
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel
```
- 문장 생성과 GPT를 만들 때 사용되는 모듈이다.
```python
model = TFGPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2", from_pt = True)
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
```
- 사전학습된 모델과 토큰화 모델을 불러왔다.

```python
sent = "프로젝트를 성공하려면 팀원들은"
input_ids = tokenizer.encode(sent)
input_ids = tf.convert_to_tensor([input_ids])
```
- 미완성된 문장을 sent에 넣은 뒤 그 문장을 토큰화 해 input_ids로 저장했다.
    - 아래 코드에서 대괄호를 지정해 넣어준 이유는 지정해주지 않는 다면 1차원 구조로 출력되는데 2차원 구조로 만들어야 하기에 []안에 값을 넣었다.
```python
output = model.generate(input_ids,
               max_length = 128,
               repetition_penalty = 2.0,
               use_cache = True)
```
- generate는 텍스트를 생성하는 함수다.
    - 여기서 repetition_penalty의 의미는 반복을 방지하기 위해 주는 값으로 값이 높을 수록 반복을 하지 않는다는 특징이 있다.
    - use_cache는 false로 설정하면 수정된 부분만을 새로 계산하는 효율적인 처리방법을 사용하고 True로 설정하면 이전 결과를 그대로 사용해 새로운 입력에 대해 새로 계산하지 않는다.

```python
output_ids = output.numpy().tolist()[0]
```
- tokenizer의 디코드 함수로 전달하기 위해 단일 리스트 구조로 변경하는 과정이다.

```python
tokenizer.decode(output_ids)
```
- 프로젝트를 성공하려면 팀원들은 물론 직원들도 함께 노력해야 한다.\n이런 점에서 이번 행사는 ‘팀원들과 소통하는 자리’라는 의미를 담고 있다.\n특히 올해는 지난해보다 더 많은 직원이 참석할 것으로 예상돼 더욱 의미가 크다.\n올해로 3회째를 맞는 이 행사에는 삼성전자, LG전자 등 국내 주요 전자업체 임직원 약 1만명이 참가한다.\n삼성 관계자는 “지난해에 비해 규모가 커진 만큼 다양한 프로그램을 준비했다”며 “이번 행사를 통해 직원들의 사기를 진작하고 회사의 비전을 공유하기 위해 노력하겠다”고 말했다.</d> 한국거래소는 오는 30일부터 다음달 2일까지 코스피200선물 야간옵션시장 개설
    - 이라는 값이 나왔다.
```python
sent = "팀원으로서 내가 해야 할 작업은 어떻게 정하나요?"
input_ids = tokenizer.encode(sent)
input_ids = tf.convert_to_tensor([input_ids])
output = model.generate(input_ids,
               max_length = 128,
               repetition_penalty = 1.0,
               use_cache = True)
output_ids = output.numpy().tolist()[0]
tokenizer.decode(output_ids)
```
- 같은 코드를 repetition_penalty만 값을 달리 줘 출력했다.