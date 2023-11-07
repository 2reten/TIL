# CNN

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.image import pad_to_bounding_box
from tensorflow.image import central_crop
from tensorflow.image import resize
```
- 이미지 데이터를 분류하는 모델인 CNN을 배웠다.

```python
bgd = image.load_img('dogs.png')
image.img_to_array(bgd).shape # 높이, 너비, 채널
bgd_vector = np.asarray(image.img_to_array(bgd))
```
- 강아지의 사진을 불러와서 bgd라는 변수에 저장하고 shape를 확인했다.
    - 그 후 bgd_vector라는 변수에 np.asarray로 변환한 값을 저장했다.

```python
bgd_vector = bgd_vector/255
plt.imshow(bgd_vector)
plt.show()
```
- 이미지의 정규화 방법이다 이미지는 최대값이 255이므로 255로 나누면 정규화가 된다.

```python
target_height = 4500
target_width = 4500
 
source_height = bgd_vector.shape[0]
source_width = bgd_vector.shape[1]
 
bgd_vector_pad = pad_to_bounding_box(bgd_vector, 
                                     int((target_height-source_height)/2), 
                                     int((target_width-source_width)/2), 
                                     target_height, 
                                     target_width)

bgd_vector_pad.shape

plt.imshow(bgd_vector_pad)
plt.show()

image.save_img('dogs_pad.png', bgd_vector_pad)
```
- 이미지의 변경할 크기를 설정하고 현재 이미지의 크기를 지정해줬다.
    - 그리고 padding을 이용해서 이미지의 크기를 변환시킨 뒤 저장했다.
```python
bgd_vector_crop = central_crop(bgd_vector, .5)
 
bgd_vector_crop.shape
 
plt.imshow(bgd_vector_crop)
plt.show()
```
- 가운데를 중심으로 50%만 crop시켰다.

```python
bgd_vector_resize = resize(bgd_vector, (300,300))
 
bgd_vector_resize.shape
 
plt.imshow(bgd_vector_resize)
```
- 300,300으로 사이즈를 변경했다.

```python
from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet', include_top=True)
model.summary()
```
- 사전학습 모델을 불러왔다. (케라스에서 클래스 형태로 제공된다.)
```python
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
 
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
 
img = Image.open('dogs.png')
```
- 필요한 모듈을 다시 import 한 뒤 강아지 사진을 다시 불러왔다.

```python
w, h = img.size
s = min(w, h)
y = (h - s) // 2
x = (w - s) // 2
print(w, h, x, y, s) # 1148 1157 0 4 1148
img = img.crop((x, y, x+s, y+s))
```
```python
model.layers[0].input_shape
```
- [(None, 224, 224, 3)]으로 출력됐다.
```python
target_size = 224
img = img.resize((target_size, target_size))
```
- 이미지의 사이즈를 224로 재조정했다.

```python
np_img = image.img_to_array(img)
np_img.shape  #(224, 224, 3)
img_batch = np.expand_dims(np_img, axis=0)
img_batch.shape # (1, 224, 224, 3)
```
- expand를 이용해 차원을 추가했다. 1의 의미는 사진의 장 수다.

```python
pre_processed = preprocess_input(img_batch)
y_preds = model.predict(pre_processed)
 
y_preds.shape  # 종속변수가 취할 수 있는 값의 수 = 1000
 
np.set_printoptions(suppress=True, precision=10)
y_preds
 
#가장 확률이 높은 값
np.max(y_preds)
1/1 [==============================] - 0s 469ms/step
0.78416103
```
- 모델 예측 코드다. 여러가지 값을 가져와 정확도를 출력한다.
```python
np.sum(y_preds)
```
- sum 값은 당연히도 1이 나온다.


```python
decode_predictions(y_preds, top=10)
35363/35363 [==============================] - 0s 1us/step
[[('n02113624', 'toy_poodle', 0.78416103),
  ('n02113712', 'miniature_poodle', 0.16782114),
  ('n02096437', 'Dandie_Dinmont', 0.01923898),
  ('n02093647', 'Bedlington_terrier', 0.016176438),
  ('n02113799', 'standard_poodle', 0.012043098),
  ('n02097047', 'miniature_schnauzer', 8.5362444e-05),
  ('n02105641', 'Old_English_sheepdog', 8.354024e-05),
  ('n02086240', 'Shih-Tzu', 7.7943914e-05),
  ('n02085936', 'Maltese_dog', 1.9853807e-05),
  ('n02106382', 'Bouvier_des_Flandres', 1.9203428e-05)]]
```
- toy_poodle이 가장 높은 값을 가졌다.

```python
image = Image.open('dogs.png').resize((500, 400))
image_tensor = tf.keras.preprocessing.image.img_to_array(image)
flip_lr_tensor = tf.image.flip_left_right(image_tensor)
flip_ud_tensor = tf.image.flip_up_down(image_tensor)
flip_lr_image = tf.keras.preprocessing.image.array_to_img(flip_lr_tensor)
flip_ud_image = tf.keras.preprocessing.image.array_to_img(flip_ud_tensor)

plt.figure(figsize=(12,12))

plt.subplot(1,3,1)
plt.title('Original image')
plt.imshow(image)

plt.subplot(1,3,2)
plt.title('flip_left_right')
plt.imshow(flip_lr_image)

plt.subplot(1,3,3)
plt.title('flip_up_down')
plt.imshow(flip_ud_image)

plt.show()
```
- 이미지를 좌우반전, 상하반전을 시키는 코드다.

```python
plt.figure(figsize=(12,16))

row = 4
for i in range(row):
  flip_lr_tensor = tf.image.random_flip_left_right(image_tensor)
  flip_ud_tensor = tf.image.random_flip_up_down(image_tensor)
  flip_lr_image = tf.keras.preprocessing.image.array_to_img(flip_lr_tensor)
  flip_ud_image = tf.keras.preprocessing.image.array_to_img(flip_ud_tensor)

  plt.subplot(4,3, i*3+1)
  plt.title('Original image')
  plt.imshow(image)

  plt.subplot(4,3,i*3+2)
  plt.title('flip_left_right')
  plt.imshow(flip_lr_image)

  plt.subplot(4,3,i*3+3)
  plt.title('flip_up_down')
  plt.imshow(flip_ud_image)
```
- 이 코드를 이용해서 좌우반전, 상하반전을 시키고 만약 이 값을 저장해 데이터에 추가한다면 모델의 정확도를 올릴수가 있다.

```python
plt.figure(figsize=(12,15))

central_fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
col = len(central_fractions)
for i, frac in enumerate(central_fractions):
  cropped_tensor = tf.image.central_crop(image_tensor, frac)
  cropped_img = tf.keras.preprocessing.image.array_to_img(cropped_tensor)

  plt.subplot(1, col+1, i+1)
  plt.title(f'Center crop: {frac}')
  plt.imshow(cropped_img)
```
- 이미지의 랜덤 부분을 확대해 출력하는 코드다.

```python
plt.figure(figsize=(12, 15))

random_bright_tensor = tf.image.random_brightness(image_tensor, max_delta=128)
random_bright_tensor = tf.clip_by_value(random_bright_tensor, 0, 255)
random_bright_image = tf.keras.preprocessing.image.array_to_img(random_bright_tensor)

plt.imshow(random_bright_image)
plt.show()
```
- 이미지의 밝기조절을 하는 코드다.

```python
plt.figure(figsize=(12,15))

for i in range(5):
  random_bright_tensor = tf.image.random_brightness(image_tensor, max_delta=128)
  random_bright_tensor = tf.clip_by_value(random_bright_tensor, 0, 255)
  random_bright_image = tf.keras.preprocessing.image.array_to_img(random_bright_tensor)

  plt.subplot(1, 5, i+1)
  plt.imshow(random_bright_image)
```
- 랜덤값으로 이미지의 밝기를 조절해 출력해봤다.

```python
import numpy as np
image = Image.open("dogs.png").resize((400, 300)) # 이미지에 따라 숫자를 바꾸어 보세요.
image_arr = np.array(image)
image_arr.shape

import albumentations as A

for i in range(10):
    transform = A.Compose([
      A.Affine(rotate=(-45,45), scale=(0.5,0.9), p=0.5)
    ])
    transformed = transform(image=image_arr)
    plt.figure(figsize=(12,12))
    plt.imshow((transformed['image']))
    plt.show()
```
- 이미지의 각도를 돌리는 코드다. 이 역시 데이터를 asarray화 해서 저장한다면 정확도를 올리는데 도움이 된다.