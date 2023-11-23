```python
# 내가 가지고 있는 이미지 데이터를 학습시켜서 새로운 이미지 데이터가 입력되었을 때, 대상을 추출하는 YoLo 알고리즘을 구현한다.
# custom dataset -> pretrained model(YoLo)을 불러와 파인튜닝
```

```python
!pip install PyYAML
pip install ultralytics
import zipfile
import yaml
import ultralytics
from ultralytics import YOLO
from glob import glob
import numpy as np
```

```python
!wget -O Aquarium_Data.zip https://public.roboflow.com/ds/DCRZtT8lyS?key=1t9V07QK0j
```
- 사이트내에 있는 데이터를 불러와 저장하는 코드다.

```python
with zipfile.ZipFile("/content/Aquarium_Data.zip") as target_file:
  target_file.extractall("/content/Aquarium_Data")
```
- zipfile을 이용해서 타겟파일을 추출하는 코드다.
```python
data = {
    "train": "/content/Aquarium_Data/train/images",
    "test": "/content/Aquarium_Data/train/images",
    "val": "/content/Aquarium_Data/train/images",
    "names": ["fish", "jellyfish","penguin", "puffin", "shark", "starfish", "stingray"],
    "nc":7
}
with open("/content/Aquarium_Data/Aquarium_Data.yaml", "w") as f:
  yaml.dump(data, f)
with open("/content/Aquarium_Data/Aquarium_Data.yaml", "r") as f:
  aquarium_yaml = yaml.safe_load(f)
  display(aquarium_yaml)
```
- 데이터를 각 train,test,validation파일을 이용해서 불러오고 이름은 클래스로 지정되어있는 7가지와 종류의 수를 입력했다.
    - with구문을 이용해서 yaml파일을 읽어왔다.
```python
model = YOLO("yolov8n.pt")
```
- YOLO("사전학습된 모델명")이다.

## 파인튜닝 과정

```python
model.train(data = "/content/Aquarium_Data/Aquarium_Data.yaml", epochs = 10, patience = 5, batch = 32, imgsz = 416)
```
- 사전에 학습된 yolo모델에 내가 가지고 있는 커스텀 데이터로 추가 학습(파인튜닝)

```python
test_img = glob("/content/Aquarium_Data/test/images/*")
```
- *을 이용해서 모든 이미지를 불러왔다.

```python
test_img.sort()
results = model.predict(source = "/content/Aquarium_Data/test/images/", save = True)
```
- 학습시킨 모델로 test데이터를 이용해서 예측해봤다.
```python
for result in results:

    uniq, cnt = np.unique(result.boxes.cls.cpu().numpy(), return_counts=True)  # Torch.Tensor -> numpy
    uniq_cnt_dict = dict(zip(uniq, cnt))

    print('\n{class num:counts} =', uniq_cnt_dict,'\n')

    for c in result.boxes.cls:
        print('class num =', int(c), ', class_name =', model.names[int(c)])
```
```python
[ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[235, 233, 193],
         [234, 232, 192],
         [231, 228, 190],
         ...,
         [129, 124, 121],
         [128, 123, 122],
         [128, 123, 122]],
 
        [[232, 225, 186],
         [226, 221, 182],
         [221, 215, 178],
         ...,
         [131, 126, 123],
         [130, 125, 124],
         [130, 125, 124]],
 
        [[235, 221, 185],
         [225, 213, 177],
         [213, 201, 167],
         ...,
         [128, 123, 120],
         [128, 123, 120],
         [128, 123, 120]],
 
        ...,
 
        [[  1,   2,   0],
         [  1,   2,   0],
         [  1,   2,   0],
         ...,
         [ 35,  22,   8],
         [ 35,  22,   8],
         [ 35,  22,   8]],
 
        [[  1,   2,   0],
         [  1,   2,   0],
         [  1,   2,   0],
         ...,
         [ 35,  22,   8],
         [ 35,  22,   8],
         [ 35,  22,   8]],
 
        [[  1,   2,   0],
         [  1,   2,   0],
         [  1,   2,   0],
         ...,
         [ 35,  22,   8],
         [ 35,  22,   8],
         [ 35,  22,   8]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2289_jpeg_jpg.rf.fe2a7a149e7b11f2313f5a7b30386e85.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.289295196533203, 'inference': 13.1988525390625, 'postprocess': 2.566814422607422},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[168, 113,  98],
         [168, 113,  98],
         [169, 114,  99],
         ...,
         [226, 174, 168],
         [225, 173, 167],
         [224, 172, 166]],
 
        [[167, 112,  97],
         [167, 112,  97],
         [167, 112,  97],
         ...,
         [227, 175, 169],
         [226, 174, 168],
         [225, 173, 167]],
 
        [[167, 112,  97],
         [166, 111,  96],
         [165, 110,  95],
         ...,
         [228, 176, 169],
         [227, 175, 168],
         [227, 175, 168]],
 
        ...,
 
        [[ 26,  33,  28],
         [ 29,  36,  31],
         [ 31,  38,  33],
         ...,
         [ 98, 108,  91],
         [ 95, 105,  88],
         [ 91, 101,  84]],
 
        [[ 26,  33,  28],
         [ 31,  38,  33],
         [ 32,  39,  34],
         ...,
         [ 93, 103,  87],
         [ 93, 103,  87],
         [ 90, 100,  84]],
 
        [[ 26,  33,  28],
         [ 32,  39,  34],
         [ 33,  40,  35],
         ...,
         [ 92, 102,  86],
         [ 93, 103,  87],
         [ 90, 100,  84]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2301_jpeg_jpg.rf.2c19ae5efbd1f8611b5578125f001695.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.9652843475341797, 'inference': 15.310525894165039, 'postprocess': 2.466917037963867},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[110, 137, 163],
         [116, 143, 169],
         [127, 154, 181],
         ...,
         [167, 159, 169],
         [167, 159, 169],
         [169, 161, 171]],
 
        [[113, 140, 167],
         [119, 146, 173],
         [129, 156, 183],
         ...,
         [174, 166, 176],
         [174, 166, 176],
         [176, 168, 178]],
 
        [[113, 141, 171],
         [113, 141, 171],
         [115, 143, 173],
         ...,
         [178, 170, 180],
         [177, 171, 182],
         [178, 172, 183]],
 
        ...,
 
        [[ 59,  65,  46],
         [ 60,  66,  47],
         [ 61,  67,  48],
         ...,
         [ 76,  92,  69],
         [ 76,  92,  69],
         [ 76,  92,  69]],
 
        [[ 61,  67,  48],
         [ 62,  68,  49],
         [ 64,  70,  51],
         ...,
         [ 76,  92,  69],
         [ 76,  92,  69],
         [ 75,  91,  68]],
 
        [[ 63,  69,  50],
         [ 64,  70,  51],
         [ 65,  71,  52],
         ...,
         [ 76,  92,  69],
         [ 75,  91,  68],
         [ 74,  90,  67]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2319_jpeg_jpg.rf.6e20bf97d17b74a8948aa48776c40454.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.063274383544922, 'inference': 11.79814338684082, 'postprocess': 2.5908946990966797},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[160, 179, 152],
         [150, 169, 142],
         [140, 157, 130],
         ...,
         [142, 151, 130],
         [128, 135, 114],
         [116, 123, 102]],
 
        [[166, 185, 158],
         [157, 176, 149],
         [149, 166, 139],
         ...,
         [126, 133, 112],
         [112, 119,  98],
         [106, 113,  92]],
 
        [[158, 180, 152],
         [153, 175, 147],
         [147, 166, 139],
         ...,
         [113, 118,  97],
         [104, 109,  88],
         [102, 107,  86]],
 
        ...,
 
        [[ 66,  78,  58],
         [ 66,  78,  58],
         [ 66,  78,  58],
         ...,
         [ 83,  61,  33],
         [ 83,  61,  33],
         [ 83,  61,  33]],
 
        [[ 66,  78,  58],
         [ 66,  78,  58],
         [ 66,  78,  58],
         ...,
         [ 83,  61,  33],
         [ 83,  61,  33],
         [ 83,  61,  33]],
 
        [[ 66,  78,  58],
         [ 66,  78,  58],
         [ 66,  78,  58],
         ...,
         [ 83,  61,  33],
         [ 83,  61,  33],
         [ 83,  61,  33]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2347_jpeg_jpg.rf.7c71ac4b9301eb358cd4a832844dedcb.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8885135650634766, 'inference': 11.271476745605469, 'postprocess': 2.3696422576904297},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[192, 140, 123],
         [192, 140, 123],
         [192, 140, 123],
         ...,
         [ 93, 125, 144],
         [ 87, 118, 141],
         [ 81, 113, 136]],
 
        [[193, 141, 124],
         [192, 140, 123],
         [192, 140, 123],
         ...,
         [ 98, 130, 149],
         [ 90, 121, 144],
         [ 83, 115, 138]],
 
        [[195, 141, 124],
         [195, 141, 124],
         [194, 140, 123],
         ...,
         [ 95, 127, 146],
         [ 89, 120, 143],
         [ 84, 116, 139]],
 
        ...,
 
        [[ 35,  27,  10],
         [ 35,  27,  10],
         [ 35,  27,  10],
         ...,
         [ 15,  31,  13],
         [ 14,  30,  12],
         [ 13,  29,  11]],
 
        [[ 35,  27,  10],
         [ 35,  27,  10],
         [ 34,  26,   9],
         ...,
         [ 15,  31,  13],
         [ 15,  31,  13],
         [ 13,  29,  11]],
 
        [[ 35,  27,  10],
         [ 34,  26,   9],
         [ 33,  25,   8],
         ...,
         [ 18,  34,  16],
         [ 17,  33,  15],
         [ 16,  32,  14]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2354_jpeg_jpg.rf.396e872c7fb0a95e911806986995ee7a.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8200874328613281, 'inference': 13.665199279785156, 'postprocess': 1.6057491302490234},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[190, 123, 160],
         [196, 129, 166],
         [200, 130, 170],
         ...,
         [158,  86,  79],
         [173, 102,  98],
         [174, 103,  99]],
 
        [[178, 111, 148],
         [182, 115, 152],
         [189, 119, 159],
         ...,
         [154,  84,  77],
         [167,  96,  92],
         [163,  95,  90]],
 
        [[181, 115, 150],
         [179, 113, 148],
         [185, 116, 153],
         ...,
         [151,  83,  78],
         [158,  89,  86],
         [151,  84,  81]],
 
        ...,
 
        [[163,  82,  61],
         [150,  71,  50],
         [139,  60,  39],
         ...,
         [  3,   1,   0],
         [  4,   2,   1],
         [  5,   3,   2]],
 
        [[162,  84,  61],
         [148,  72,  49],
         [137,  61,  39],
         ...,
         [  3,   1,   0],
         [  4,   2,   1],
         [  5,   3,   2]],
 
        [[165,  89,  66],
         [153,  77,  54],
         [141,  65,  43],
         ...,
         [  3,   1,   0],
         [  4,   2,   1],
         [  5,   3,   2]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2371_jpeg_jpg.rf.54505f60b6706da151c164188c305849.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8839836120605469, 'inference': 12.521982192993164, 'postprocess': 2.5589466094970703},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[52, 30, 32],
         [52, 30, 32],
         [53, 31, 33],
         ...,
         [61, 19,  6],
         [62, 21,  6],
         [63, 22,  7]],
 
        [[54, 32, 34],
         [53, 31, 33],
         [53, 31, 33],
         ...,
         [62, 20,  7],
         [60, 19,  4],
         [59, 18,  3]],
 
        [[56, 35, 37],
         [54, 33, 35],
         [52, 31, 33],
         ...,
         [62, 20,  7],
         [59, 17,  4],
         [56, 14,  1]],
 
        ...,
 
        [[31, 16, 14],
         [31, 16, 14],
         [31, 16, 14],
         ...,
         [47, 11,  3],
         [47, 11,  3],
         [46, 10,  2]],
 
        [[31, 16, 14],
         [31, 16, 14],
         [31, 16, 14],
         ...,
         [48, 12,  4],
         [48, 12,  4],
         [47, 11,  3]],
 
        [[31, 16, 14],
         [31, 16, 14],
         [31, 16, 14],
         ...,
         [49, 13,  5],
         [49, 13,  5],
         [49, 13,  5]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2379_jpeg_jpg.rf.7dc3160c937072d26d4624c6c48e904d.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8787384033203125, 'inference': 12.110471725463867, 'postprocess': 2.5038719177246094},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[255, 255, 251],
         [248, 250, 244],
         [247, 254, 249],
         ...,
         [ 44,  11,   8],
         [ 44,  11,   8],
         [ 44,  11,   8]],
 
        [[252, 255, 251],
         [249, 255, 250],
         [248, 255, 254],
         ...,
         [ 44,  11,   8],
         [ 44,  11,   8],
         [ 44,  11,   8]],
 
        [[241, 249, 248],
         [242, 253, 251],
         [243, 255, 255],
         ...,
         [ 44,  11,   8],
         [ 44,  11,   8],
         [ 44,  11,   8]],
 
        ...,
 
        [[ 40,  26,  27],
         [ 33,  19,  20],
         [ 29,  15,  16],
         ...,
         [ 63,  20,  11],
         [ 66,  23,  14],
         [ 68,  25,  16]],
 
        [[ 33,  19,  20],
         [ 30,  16,  17],
         [ 29,  15,  16],
         ...,
         [ 59,  16,   7],
         [ 63,  20,  11],
         [ 66,  23,  14]],
 
        [[ 26,  12,  13],
         [ 26,  12,  13],
         [ 29,  15,  16],
         ...,
         [ 56,  13,   4],
         [ 61,  18,   9],
         [ 65,  22,  13]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2380_jpeg_jpg.rf.a23809682eb1466c1136ca0f55de8fb5.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8870830535888672, 'inference': 11.05189323425293, 'postprocess': 2.395153045654297},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[102,  48,  31],
         [ 99,  45,  28],
         [ 94,  42,  26],
         ...,
         [ 98,  38,  26],
         [ 98,  38,  26],
         [ 97,  37,  25]],
 
        [[104,  50,  33],
         [ 99,  45,  28],
         [ 96,  41,  26],
         ...,
         [ 93,  33,  21],
         [ 95,  35,  23],
         [ 96,  36,  24]],
 
        [[109,  52,  37],
         [100,  45,  30],
         [ 96,  41,  28],
         ...,
         [ 87,  27,  15],
         [ 91,  31,  19],
         [ 94,  34,  22]],
 
        ...,
 
        [[ 15,   2,   0],
         [ 15,   2,   0],
         [ 15,   2,   0],
         ...,
         [135,  64,  67],
         [141,  64,  71],
         [155,  77,  84]],
 
        [[ 15,   2,   0],
         [ 15,   2,   0],
         [ 15,   2,   0],
         ...,
         [139,  62,  69],
         [147,  64,  73],
         [163,  77,  87]],
 
        [[ 15,   2,   0],
         [ 15,   2,   0],
         [ 15,   2,   0],
         ...,
         [140,  61,  70],
         [150,  63,  73],
         [168,  79,  89]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2387_jpeg_jpg.rf.09b38bacfab0922a3a6b66480f01b719.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7833709716796875, 'inference': 10.870695114135742, 'postprocess': 2.241373062133789},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[123, 102, 104],
         [125, 104, 106],
         [128, 107, 109],
         ...,
         [102,  74,  63],
         [102,  74,  63],
         [102,  74,  63]],
 
        [[123, 102, 104],
         [124, 103, 105],
         [126, 105, 107],
         ...,
         [104,  76,  65],
         [103,  75,  64],
         [103,  75,  64]],
 
        [[123, 102, 104],
         [123, 102, 104],
         [124, 103, 105],
         ...,
         [106,  78,  67],
         [105,  77,  66],
         [105,  77,  66]],
 
        ...,
 
        [[158,  75,  37],
         [156,  73,  35],
         [157,  72,  34],
         ...,
         [207, 156, 140],
         [249, 194, 181],
         [222, 167, 152]],
 
        [[161,  77,  41],
         [158,  74,  38],
         [175,  89,  53],
         ...,
         [148,  94,  77],
         [162, 103,  88],
         [149,  89,  73]],
 
        [[159,  75,  39],
         [165,  81,  45],
         [204, 118,  82],
         ...,
         [135,  76,  60],
         [143,  83,  67],
         [149,  87,  71]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2395_jpeg_jpg.rf.9f1503ad3b7a7c7938daed057cc4e9bc.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7282962799072266, 'inference': 10.614633560180664, 'postprocess': 2.2475719451904297},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 68,  21,   0],
         [ 68,  21,   0],
         [ 68,  21,   0],
         ...,
         [157,  71,  41],
         [157,  71,  41],
         [156,  70,  40]],
 
        [[ 68,  21,   0],
         [ 68,  21,   0],
         [ 68,  21,   0],
         ...,
         [157,  71,  41],
         [159,  71,  41],
         [156,  70,  40]],
 
        [[ 69,  22,   0],
         [ 69,  22,   0],
         [ 69,  22,   0],
         ...,
         [159,  70,  43],
         [161,  70,  43],
         [158,  69,  42]],
 
        ...,
 
        [[156, 103,  76],
         [157, 104,  77],
         [155, 105,  77],
         ...,
         [200,  89,  57],
         [198,  87,  55],
         [196,  85,  53]],
 
        [[154, 100,  75],
         [152,  98,  73],
         [149,  98,  72],
         ...,
         [195,  84,  52],
         [195,  84,  52],
         [194,  83,  51]],
 
        [[156, 102,  77],
         [152,  98,  73],
         [146,  95,  69],
         ...,
         [188,  77,  45],
         [189,  78,  46],
         [190,  79,  47]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2423_jpeg_jpg.rf.1c0901882e71d5ebd26f036f4e22da65.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.814126968383789, 'inference': 11.076688766479492, 'postprocess': 2.3887157440185547},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 93,  70,  75],
         [ 93,  70,  75],
         [ 94,  71,  76],
         ...,
         [ 86,  47,  38],
         [ 86,  47,  38],
         [ 86,  47,  38]],
 
        [[ 92,  69,  74],
         [ 93,  70,  75],
         [ 93,  70,  75],
         ...,
         [ 86,  47,  38],
         [ 86,  47,  38],
         [ 86,  47,  38]],
 
        [[ 91,  68,  73],
         [ 92,  69,  74],
         [ 92,  69,  74],
         ...,
         [ 87,  48,  39],
         [ 87,  48,  39],
         [ 87,  48,  39]],
 
        ...,
 
        [[117,  56,  46],
         [ 98,  42,  31],
         [ 75,  25,  13],
         ...,
         [ 69,  22,   0],
         [ 69,  22,   0],
         [ 69,  22,   0]],
 
        [[111,  47,  36],
         [ 90,  30,  18],
         [ 70,  17,   4],
         ...,
         [ 69,  22,   0],
         [ 69,  22,   0],
         [ 69,  22,   0]],
 
        [[106,  40,  29],
         [ 82,  20,   9],
         [ 66,  11,   0],
         ...,
         [ 69,  22,   0],
         [ 69,  22,   0],
         [ 69,  22,   0]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2434_jpeg_jpg.rf.8b20d3270d4fbc497c64125273f46ecb.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8749237060546875, 'inference': 13.692140579223633, 'postprocess': 2.7425289154052734},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 87,  32,   1],
         [ 87,  32,   1],
         [ 87,  30,   4],
         ...,
         [160,  68,  43],
         [158,  66,  41],
         [158,  66,  41]],
 
        [[ 89,  31,   2],
         [ 89,  31,   2],
         [ 89,  30,   4],
         ...,
         [159,  67,  42],
         [158,  66,  41],
         [158,  66,  41]],
 
        [[ 91,  30,   4],
         [ 91,  30,   4],
         [ 91,  29,   5],
         ...,
         [158,  66,  41],
         [158,  66,  41],
         [158,  66,  41]],
 
        ...,
 
        [[154, 111,  84],
         [154, 111,  84],
         [154, 111,  84],
         ...,
         [141, 101,  73],
         [140, 100,  72],
         [140, 100,  72]],
 
        [[155, 112,  85],
         [154, 111,  84],
         [155, 112,  85],
         ...,
         [141, 101,  73],
         [141, 101,  73],
         [140, 100,  72]],
 
        [[157, 114,  87],
         [156, 113,  86],
         [156, 113,  86],
         ...,
         [141, 101,  73],
         [141, 101,  73],
         [140, 100,  72]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2446_jpeg_jpg.rf.06ee05e92df8e3c33073147d8f595211.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.9505023956298828, 'inference': 10.309219360351562, 'postprocess': 2.319812774658203},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 79,  30,  14],
         [ 78,  29,  13],
         [ 78,  29,  13],
         ...,
         [123,  52,  32],
         [123,  52,  32],
         [123,  52,  32]],
 
        [[ 77,  28,  12],
         [ 77,  28,  12],
         [ 77,  28,  12],
         ...,
         [123,  52,  32],
         [123,  52,  32],
         [123,  52,  32]],
 
        [[ 76,  27,  11],
         [ 76,  27,  11],
         [ 75,  26,  10],
         ...,
         [123,  52,  32],
         [123,  52,  32],
         [123,  52,  32]],
 
        ...,
 
        [[173, 126,  95],
         [173, 126,  95],
         [172, 125,  94],
         ...,
         [104,  73,  48],
         [105,  74,  49],
         [104,  73,  48]],
 
        [[176, 126,  96],
         [176, 126,  96],
         [176, 126,  96],
         ...,
         [106,  75,  50],
         [106,  75,  50],
         [105,  74,  49]],
 
        [[174, 124,  94],
         [175, 125,  95],
         [175, 125,  95],
         ...,
         [107,  76,  51],
         [106,  75,  50],
         [105,  74,  49]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2448_jpeg_jpg.rf.28ce79dab47ad525751d5407be09bc3d.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.669095993041992, 'inference': 10.468244552612305, 'postprocess': 2.1533966064453125},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 87,  26,  12],
         [ 87,  28,  13],
         [ 89,  29,  17],
         ...,
         [123,  57,  38],
         [124,  58,  39],
         [132,  66,  47]],
 
        [[ 92,  31,  17],
         [ 92,  33,  18],
         [ 91,  33,  21],
         ...,
         [126,  60,  41],
         [129,  63,  44],
         [136,  70,  51]],
 
        [[ 89,  30,  15],
         [ 91,  32,  17],
         [ 91,  33,  21],
         ...,
         [128,  62,  43],
         [130,  64,  45],
         [132,  66,  47]],
 
        ...,
 
        [[155, 105,  75],
         [154, 104,  74],
         [152, 102,  72],
         ...,
         [143, 166, 138],
         [148, 172, 142],
         [158, 182, 152]],
 
        [[157, 107,  77],
         [156, 106,  76],
         [154, 104,  74],
         ...,
         [139, 157, 126],
         [144, 163, 130],
         [156, 175, 142]],
 
        [[159, 109,  79],
         [158, 108,  78],
         [156, 106,  76],
         ...,
         [140, 157, 124],
         [150, 165, 133],
         [167, 183, 149]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2450_jpeg_jpg.rf.ff673921373de3bfc275863e3befeefe.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.6052722930908203, 'inference': 12.141227722167969, 'postprocess': 2.392292022705078},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[233,  49,   3],
         [233,  49,   3],
         [233,  49,   3],
         ...,
         [217,  43,   0],
         [217,  43,   0],
         [217,  43,   0]],
 
        [[233,  49,   3],
         [233,  49,   3],
         [233,  49,   3],
         ...,
         [217,  43,   0],
         [217,  43,   0],
         [217,  43,   0]],
 
        [[233,  49,   1],
         [233,  49,   1],
         [233,  49,   1],
         ...,
         [217,  43,   0],
         [217,  43,   0],
         [217,  43,   0]],
 
        ...,
 
        [[ 14, 135, 154],
         [ 11, 132, 151],
         [ 12, 132, 151],
         ...,
         [ 51,  15,   5],
         [ 51,  15,   5],
         [ 51,  15,   5]],
 
        [[ 12, 133, 149],
         [  9, 130, 146],
         [ 11, 129, 146],
         ...,
         [ 51,  15,   5],
         [ 51,  15,   5],
         [ 51,  15,   5]],
 
        [[ 10, 131, 147],
         [  9, 130, 146],
         [ 10, 128, 145],
         ...,
         [ 51,  15,   5],
         [ 51,  15,   5],
         [ 51,  15,   5]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2465_jpeg_jpg.rf.7e699ec1d2e373d93dac32cd02db9438.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.005338668823242, 'inference': 10.379552841186523, 'postprocess': 2.2110939025878906},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[241,  57,   0],
         [241,  57,   0],
         [241,  57,   0],
         ...,
         [130,   2,   3],
         [130,   2,   3],
         [130,   2,   3]],
 
        [[241,  57,   0],
         [241,  57,   0],
         [241,  57,   0],
         ...,
         [130,   2,   3],
         [130,   2,   3],
         [130,   2,   3]],
 
        [[241,  57,   0],
         [241,  57,   0],
         [241,  57,   0],
         ...,
         [130,   2,   3],
         [130,   2,   3],
         [130,   2,   3]],
 
        ...,
 
        [[ 65,  21,   8],
         [ 65,  21,   8],
         [ 65,  21,   8],
         ...,
         [ 32,   2,   1],
         [ 33,   3,   2],
         [ 34,   4,   3]],
 
        [[ 65,  21,   8],
         [ 65,  21,   8],
         [ 65,  21,   8],
         ...,
         [ 33,   3,   2],
         [ 34,   4,   3],
         [ 34,   4,   3]],
 
        [[ 65,  21,   8],
         [ 65,  21,   8],
         [ 65,  21,   8],
         ...,
         [ 34,   4,   3],
         [ 34,   4,   3],
         [ 34,   4,   3]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2466_jpeg_jpg.rf.53886abb9947ec4e47405957b30fe314.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7714500427246094, 'inference': 9.886741638183594, 'postprocess': 2.2449493408203125},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[36,  5,  2],
         [36,  5,  2],
         [36,  5,  2],
         ...,
         [52,  2,  2],
         [52,  2,  2],
         [52,  2,  2]],
 
        [[36,  5,  2],
         [36,  5,  2],
         [36,  5,  2],
         ...,
         [49,  4,  1],
         [49,  3,  2],
         [49,  4,  1]],
 
        [[36,  5,  2],
         [36,  5,  2],
         [36,  5,  2],
         ...,
         [40,  6,  0],
         [40,  5,  1],
         [40,  6,  0]],
 
        ...,
 
        [[72,  8,  3],
         [72,  8,  3],
         [72,  8,  3],
         ...,
         [97,  9,  2],
         [97,  9,  2],
         [97,  9,  2]],
 
        [[72,  8,  3],
         [72,  8,  3],
         [72,  8,  3],
         ...,
         [99,  9,  2],
         [99,  9,  2],
         [99,  9,  2]],
 
        [[72,  8,  3],
         [72,  8,  3],
         [72,  8,  3],
         ...,
         [99,  9,  2],
         [99,  9,  2],
         [99,  9,  2]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2468_jpeg_jpg.rf.c933cc14c99b11a90413a1490d4556db.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8062591552734375, 'inference': 10.141849517822266, 'postprocess': 2.313852310180664},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 37,   6,   3],
         [ 37,   6,   3],
         [ 37,   6,   3],
         ...,
         [252,  66,   0],
         [255,  69,   0],
         [255,  71,   0]],
 
        [[ 37,   6,   3],
         [ 37,   6,   3],
         [ 37,   6,   3],
         ...,
         [252,  66,   0],
         [255,  69,   0],
         [255,  71,   0]],
 
        [[ 37,   6,   3],
         [ 37,   6,   3],
         [ 37,   6,   3],
         ...,
         [253,  67,   0],
         [255,  69,   0],
         [255,  70,   0]],
 
        ...,
 
        [[ 17,   1,   2],
         [ 17,   1,   2],
         [ 15,   1,   2],
         ...,
         [ 68,  19,   5],
         [ 68,  19,   5],
         [ 68,  19,   5]],
 
        [[ 17,   1,   2],
         [ 17,   1,   2],
         [ 15,   1,   2],
         ...,
         [ 69,  20,   6],
         [ 69,  20,   6],
         [ 70,  21,   7]],
 
        [[ 17,   1,   2],
         [ 17,   1,   2],
         [ 15,   1,   2],
         ...,
         [ 70,  21,   7],
         [ 70,  21,   7],
         [ 71,  22,   8]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2470_jpeg_jpg.rf.75b359c8baa6866bfecf07a0e4e8c33d.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8362998962402344, 'inference': 10.405302047729492, 'postprocess': 2.538442611694336},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[220,  49,   3],
         [220,  49,   3],
         [223,  49,   3],
         ...,
         [255,  61,   1],
         [255,  61,   1],
         [255,  61,   1]],
 
        [[222,  50,   2],
         [223,  49,   2],
         [225,  49,   2],
         ...,
         [255,  61,   1],
         [255,  61,   1],
         [255,  61,   1]],
 
        [[227,  50,   0],
         [227,  50,   0],
         [229,  50,   0],
         ...,
         [255,  61,   1],
         [255,  61,   1],
         [255,  61,   1]],
 
        ...,
 
        [[ 80,  28,  11],
         [ 80,  28,  11],
         [ 80,  28,  11],
         ...,
         [ 83,  22,   2],
         [ 83,  22,   2],
         [ 83,  22,   2]],
 
        [[ 80,  28,  11],
         [ 80,  28,  11],
         [ 80,  28,  11],
         ...,
         [ 83,  22,   2],
         [ 83,  22,   2],
         [ 83,  22,   2]],
 
        [[ 80,  28,  11],
         [ 80,  28,  11],
         [ 80,  28,  11],
         ...,
         [ 83,  22,   2],
         [ 83,  22,   2],
         [ 83,  22,   2]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2473_jpeg_jpg.rf.6284677f9c781b0cfeec54981a17d573.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8062591552734375, 'inference': 13.627290725708008, 'postprocess': 2.4454593658447266},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[234,  82,  11],
         [218,  85,  23],
         [240, 142, 102],
         ...,
         [255,  93,   0],
         [255,  93,   0],
         [255,  93,   0]],
 
        [[237,  85,  14],
         [219,  86,  24],
         [236, 141,  98],
         ...,
         [255,  93,   0],
         [255,  93,   0],
         [255,  93,   0]],
 
        [[237,  88,  14],
         [218,  87,  24],
         [234, 140,  97],
         ...,
         [255,  93,   0],
         [255,  93,   0],
         [255,  93,   0]],
 
        ...,
 
        [[ 77,  83,  88],
         [ 65,  70,  73],
         [ 53,  52,  54],
         ...,
         [ 71,  28,  13],
         [ 70,  27,  12],
         [ 68,  25,  10]],
 
        [[ 68,  69,  73],
         [ 56,  55,  57],
         [ 47,  41,  42],
         ...,
         [ 70,  27,  12],
         [ 71,  26,  12],
         [ 72,  27,  13]],
 
        [[ 59,  58,  60],
         [ 49,  47,  47],
         [ 43,  35,  36],
         ...,
         [ 69,  26,  11],
         [ 71,  26,  12],
         [ 74,  29,  15]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2477_jpeg_jpg.rf.7b2692f142d53c16ad477065f1f8ae6d.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.9998550415039062, 'inference': 12.157917022705078, 'postprocess': 2.5801658630371094},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 99,  36,  22],
         [ 99,  36,  22],
         [100,  37,  23],
         ...,
         [227, 217, 187],
         [227, 205, 180],
         [182, 155, 134]],
 
        [[100,  37,  23],
         [100,  37,  23],
         [101,  38,  24],
         ...,
         [231, 220, 190],
         [236, 212, 188],
         [190, 161, 140]],
 
        [[101,  38,  24],
         [101,  38,  24],
         [102,  39,  25],
         ...,
         [244, 229, 197],
         [243, 213, 188],
         [184, 150, 127]],
 
        ...,
 
        [[142, 108,  55],
         [141, 109,  56],
         [142, 110,  57],
         ...,
         [255, 176,  97],
         [250, 175,  96],
         [247, 174,  94]],
 
        [[140, 109,  54],
         [142, 111,  56],
         [142, 112,  57],
         ...,
         [255, 162,  86],
         [255, 164,  89],
         [255, 165,  87]],
 
        [[140, 109,  54],
         [140, 110,  55],
         [142, 112,  57],
         ...,
         [247, 140,  66],
         [254, 142,  66],
         [255, 141,  66]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2496_jpeg_jpg.rf.3f91e7f18502074c89fa720a11926fab.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.0329952239990234, 'inference': 12.244462966918945, 'postprocess': 2.5823116302490234},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[229, 155,  49],
         [250, 178,  67],
         [255, 190,  74],
         ...,
         [136,  67,  40],
         [142,  69,  41],
         [141,  68,  40]],
 
        [[228, 150,  44],
         [236, 159,  49],
         [251, 174,  58],
         ...,
         [136,  69,  42],
         [136,  67,  40],
         [131,  60,  33]],
 
        [[234, 150,  44],
         [228, 145,  36],
         [236, 155,  40],
         ...,
         [129,  65,  41],
         [130,  63,  40],
         [126,  60,  35]],
 
        ...,
 
        [[249, 187, 103],
         [253, 191, 107],
         [251, 189, 105],
         ...,
         [ 66,  49,  28],
         [ 50,  33,  12],
         [ 37,  20,   0]],
 
        [[250, 188, 104],
         [248, 186, 102],
         [250, 188, 104],
         ...,
         [ 79,  60,  39],
         [ 66,  47,  26],
         [ 52,  33,  12]],
 
        [[249, 187, 103],
         [240, 178,  94],
         [246, 184, 100],
         ...,
         [ 65,  46,  25],
         [ 69,  50,  29],
         [ 69,  50,  29]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2499_jpeg_jpg.rf.6cbab3719b9063388b5ab3ab826d7bd3.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.756429672241211, 'inference': 10.760068893432617, 'postprocess': 2.4595260620117188},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[106,  99,  74],
         [108, 101,  76],
         [111, 104,  79],
         ...,
         [182, 119,  91],
         [182, 119,  91],
         [182, 119,  91]],
 
        [[100,  90,  66],
         [103,  93,  69],
         [106,  96,  72],
         ...,
         [183, 120,  92],
         [183, 120,  92],
         [183, 120,  92]],
 
        [[ 92,  80,  56],
         [ 95,  83,  59],
         [100,  88,  64],
         ...,
         [184, 121,  93],
         [184, 121,  93],
         [184, 121,  93]],
 
        ...,
 
        [[ 81,  59,  31],
         [ 78,  56,  28],
         [ 75,  53,  25],
         ...,
         [146,  98,  70],
         [146,  98,  70],
         [146,  98,  70]],
 
        [[ 83,  61,  33],
         [ 79,  57,  29],
         [ 75,  53,  25],
         ...,
         [146,  98,  70],
         [148,  98,  70],
         [148,  98,  70]],
 
        [[ 83,  61,  33],
         [ 80,  58,  30],
         [ 76,  54,  26],
         ...,
         [146,  98,  70],
         [149,  99,  71],
         [149,  99,  71]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2514_jpeg_jpg.rf.6ccb3859d75fc5cfe053b1c1474254b2.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.9359588623046875, 'inference': 11.015653610229492, 'postprocess': 2.3872852325439453},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[155, 113, 108],
         [149, 107, 102],
         [149, 107, 102],
         ...,
         [118, 109, 135],
         [116, 107, 133],
         [115, 106, 132]],
 
        [[164, 122, 117],
         [156, 116, 111],
         [156, 114, 109],
         ...,
         [121, 115, 140],
         [125, 116, 142],
         [125, 119, 144]],
 
        [[155, 114, 111],
         [150, 112, 108],
         [152, 111, 108],
         ...,
         [122, 118, 143],
         [127, 121, 146],
         [130, 126, 151]],
 
        ...,
 
        [[ 69,  93, 113],
         [ 64,  88, 108],
         [ 61,  85, 105],
         ...,
         [197, 234, 248],
         [169, 206, 220],
         [151, 188, 202]],
 
        [[ 70,  91, 113],
         [ 73,  94, 116],
         [ 78,  99, 120],
         ...,
         [193, 227, 240],
         [175, 209, 222],
         [148, 183, 196]],
 
        [[ 58,  79, 101],
         [ 66,  87, 109],
         [ 78,  97, 118],
         ...,
         [165, 197, 210],
         [156, 190, 203],
         [141, 175, 188]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2526_jpeg_jpg.rf.003e1d1d41bcd204df731b85cea68781.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.760244369506836, 'inference': 10.622978210449219, 'postprocess': 1.3675689697265625},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 77,  63,  67],
         [ 76,  62,  66],
         [ 73,  59,  63],
         ...,
         [119,  66,  56],
         [114,  61,  51],
         [113,  60,  50]],
 
        [[ 74,  60,  64],
         [ 74,  60,  64],
         [ 73,  59,  63],
         ...,
         [126,  73,  63],
         [116,  63,  53],
         [110,  57,  47]],
 
        [[ 73,  58,  62],
         [ 74,  59,  63],
         [ 74,  59,  63],
         ...,
         [135,  82,  72],
         [121,  68,  58],
         [111,  58,  48]],
 
        ...,
 
        [[ 15,   5,   5],
         [ 14,   4,   4],
         [ 14,   4,   4],
         ...,
         [203, 115, 128],
         [188, 109, 118],
         [179, 105, 111]],
 
        [[ 15,   5,   5],
         [ 14,   4,   4],
         [ 14,   4,   4],
         ...,
         [194, 108, 120],
         [184, 107, 115],
         [179, 106, 114]],
 
        [[ 15,   5,   5],
         [ 14,   4,   4],
         [ 14,   4,   4],
         ...,
         [190, 104, 116],
         [185, 107, 118],
         [183, 110, 118]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2532_jpeg_jpg.rf.2afeb76e5d9372dbbd6fbc53d5b75675.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.551555633544922, 'inference': 11.08860969543457, 'postprocess': 2.2683143615722656},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[108,  56,  39],
         [107,  55,  38],
         [105,  53,  36],
         ...,
         [ 61,  21,  22],
         [ 60,  20,  25],
         [ 67,  27,  32]],
 
        [[104,  52,  35],
         [104,  52,  35],
         [103,  51,  34],
         ...,
         [ 62,  24,  24],
         [ 60,  20,  25],
         [ 59,  22,  26]],
 
        [[105,  53,  36],
         [104,  52,  35],
         [104,  52,  35],
         ...,
         [ 66,  28,  28],
         [ 61,  24,  28],
         [ 53,  18,  22]],
 
        ...,
 
        [[ 85,  33,  27],
         [ 84,  32,  26],
         [ 84,  32,  26],
         ...,
         [ 63,  83,  94],
         [ 54,  72,  83],
         [ 52,  70,  81]],
 
        [[ 84,  32,  26],
         [ 84,  32,  26],
         [ 84,  32,  26],
         ...,
         [ 62,  82,  93],
         [ 50,  68,  79],
         [ 45,  61,  73]],
 
        [[ 84,  32,  26],
         [ 84,  32,  26],
         [ 85,  33,  27],
         ...,
         [ 66,  86,  97],
         [ 50,  68,  79],
         [ 40,  56,  68]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2544_jpeg_jpg.rf.03f51bb9e1c57fb9cd62f8cbdca14e90.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.9490718841552734, 'inference': 11.078596115112305, 'postprocess': 2.5031566619873047},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 63,  53,  70],
         [ 63,  53,  70],
         [ 62,  52,  69],
         ...,
         [ 41,  78, 116],
         [ 48,  83, 126],
         [ 53,  91, 133]],
 
        [[ 63,  53,  70],
         [ 63,  53,  70],
         [ 62,  52,  69],
         ...,
         [ 51,  83, 124],
         [ 60,  94, 137],
         [ 67, 102, 145]],
 
        [[ 63,  53,  70],
         [ 63,  53,  70],
         [ 62,  52,  69],
         ...,
         [ 53,  78, 120],
         [ 58,  87, 131],
         [ 66,  95, 139]],
 
        ...,
 
        [[ 52,   4,   3],
         [ 53,   3,   3],
         [ 55,   3,   3],
         ...,
         [ 94,  95,  99],
         [ 71,  70,  74],
         [ 56,  55,  59]],
 
        [[ 53,   4,   2],
         [ 53,   4,   2],
         [ 55,   4,   2],
         ...,
         [106, 109, 113],
         [ 86,  87,  91],
         [ 69,  71,  72]],
 
        [[ 53,   4,   2],
         [ 53,   4,   2],
         [ 57,   3,   2],
         ...,
         [111, 116, 119],
         [101, 103, 104],
         [ 85,  87,  88]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2547_jpeg_jpg.rf.9406b6f1a9fad2292c4abd28f712baaf.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8475055694580078, 'inference': 11.07025146484375, 'postprocess': 1.6887187957763672},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[160,  69,  60],
         [160,  69,  60],
         [160,  69,  60],
         ...,
         [197, 130, 107],
         [196, 129, 106],
         [196, 129, 106]],
 
        [[159,  68,  59],
         [159,  68,  59],
         [159,  68,  59],
         ...,
         [199, 132, 109],
         [198, 131, 108],
         [197, 130, 107]],
 
        [[157,  66,  57],
         [157,  66,  57],
         [157,  66,  57],
         ...,
         [199, 132, 109],
         [198, 131, 108],
         [197, 130, 107]],
 
        ...,
 
        [[111,  35,   0],
         [112,  36,   0],
         [113,  37,   1],
         ...,
         [115,  43,   1],
         [116,  44,   2],
         [116,  44,   2]],
 
        [[111,  35,   0],
         [112,  36,   0],
         [113,  37,   1],
         ...,
         [115,  43,   1],
         [116,  44,   2],
         [116,  44,   2]],
 
        [[111,  35,   0],
         [112,  36,   0],
         [113,  37,   1],
         ...,
         [115,  43,   1],
         [115,  43,   1],
         [116,  44,   2]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2570_jpeg_jpg.rf.ed40900b657a5b23d92cb2d296ad2dbc.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7309188842773438, 'inference': 10.954618453979492, 'postprocess': 2.4306774139404297},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[237,  41,   0],
         [237,  41,   0],
         [237,  41,   0],
         ...,
         [229,  41,   0],
         [229,  41,   0],
         [229,  41,   0]],
 
        [[237,  41,   0],
         [237,  41,   0],
         [237,  41,   0],
         ...,
         [229,  41,   0],
         [229,  41,   0],
         [229,  41,   0]],
 
        [[237,  41,   0],
         [237,  41,   0],
         [237,  41,   0],
         ...,
         [229,  41,   0],
         [229,  41,   0],
         [229,  41,   0]],
 
        ...,
 
        [[ 53,  17,   7],
         [ 52,  16,   6],
         [ 51,  15,   5],
         ...,
         [ 66,  14,   7],
         [ 67,  15,   8],
         [ 68,  16,   9]],
 
        [[ 53,  17,   7],
         [ 52,  16,   6],
         [ 51,  15,   5],
         ...,
         [ 65,  13,   6],
         [ 66,  14,   7],
         [ 67,  15,   8]],
 
        [[ 54,  18,   8],
         [ 53,  17,   7],
         [ 51,  15,   5],
         ...,
         [ 62,  10,   3],
         [ 62,  10,   3],
         [ 63,  11,   4]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2574_jpeg_jpg.rf.ca0c3ad32384309a61e92d9a8bef87b9.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.656221389770508, 'inference': 15.549659729003906, 'postprocess': 4.001379013061523},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[131,  43,   3],
         [131,  43,   3],
         [131,  43,   3],
         ...,
         [102,  62,  20],
         [113,  73,  31],
         [118,  80,  38]],
 
        [[131,  43,   3],
         [131,  43,   3],
         [131,  43,   3],
         ...,
         [103,  61,  19],
         [112,  72,  30],
         [119,  81,  39]],
 
        [[131,  43,   3],
         [131,  43,   3],
         [131,  43,   3],
         ...,
         [ 94,  51,  12],
         [107,  64,  25],
         [116,  75,  36]],
 
        ...,
 
        [[235, 190, 103],
         [233, 186, 100],
         [233, 179,  94],
         ...,
         [216, 149,  88],
         [212, 150,  90],
         [211, 154,  93]],
 
        [[227, 181,  93],
         [228, 178,  90],
         [226, 171,  86],
         ...,
         [223, 151,  91],
         [216, 150,  91],
         [212, 148,  90]],
 
        [[218, 170,  82],
         [219, 169,  81],
         [223, 166,  81],
         ...,
         [226, 153,  93],
         [219, 150,  93],
         [211, 144,  87]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2582_jpeg_jpg.rf.14f175066ce74b470bf31fa0c7a096cd.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.1300315856933594, 'inference': 15.332460403442383, 'postprocess': 2.730846405029297},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 76,  29,  15],
         [ 76,  29,  15],
         [ 76,  29,  15],
         ...,
         [108,  36,   6],
         [109,  37,   7],
         [110,  38,   8]],
 
        [[ 76,  29,  15],
         [ 76,  29,  15],
         [ 76,  29,  15],
         ...,
         [109,  37,   7],
         [110,  38,   8],
         [111,  39,   9]],
 
        [[ 77,  30,  16],
         [ 76,  29,  15],
         [ 76,  29,  15],
         ...,
         [110,  38,   8],
         [111,  39,   9],
         [112,  40,  10]],
 
        ...,
 
        [[ 65,  31,  15],
         [ 59,  25,   9],
         [ 55,  21,   5],
         ...,
         [204, 166, 102],
         [206, 169, 101],
         [213, 177, 107]],
 
        [[ 56,  22,   6],
         [ 52,  18,   2],
         [ 52,  18,   2],
         ...,
         [211, 173, 109],
         [214, 178, 108],
         [217, 184, 111]],
 
        [[ 60,  26,  10],
         [ 57,  23,   7],
         [ 56,  22,   6],
         ...,
         [220, 182, 118],
         [219, 185, 115],
         [222, 189, 116]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2588_jpeg_jpg.rf.cb9cea8f05891cfd55a3e93f2908201f.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.0694732666015625, 'inference': 10.924339294433594, 'postprocess': 2.5920867919921875},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 33,  52,  57],
         [ 35,  54,  59],
         [ 46,  65,  70],
         ...,
         [171, 135, 125],
         [172, 136, 126],
         [172, 136, 126]],
 
        [[ 31,  50,  55],
         [ 31,  50,  55],
         [ 41,  60,  65],
         ...,
         [174, 138, 128],
         [174, 138, 128],
         [173, 137, 127]],
 
        [[ 26,  45,  50],
         [ 33,  52,  57],
         [ 45,  64,  69],
         ...,
         [178, 142, 132],
         [176, 140, 130],
         [175, 139, 129]],
 
        ...,
 
        [[ 81,  85,  80],
         [ 81,  87,  82],
         [ 72,  77,  75],
         ...,
         [ 61,  45,  32],
         [ 67,  48,  35],
         [ 76,  54,  42]],
 
        [[ 94,  91,  87],
         [ 96,  95,  91],
         [ 87,  85,  84],
         ...,
         [ 70,  50,  33],
         [ 76,  47,  32],
         [ 80,  50,  33]],
 
        [[ 98,  93,  90],
         [104,  99,  96],
         [ 94,  90,  89],
         ...,
         [ 78,  54,  36],
         [ 82,  49,  33],
         [ 83,  47,  29]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2630_jpeg_jpg.rf.310f0c986a72be46b80ce31c2d00e46d.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.1767616271972656, 'inference': 11.896848678588867, 'postprocess': 2.5000572204589844},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[103, 111, 118],
         [ 94, 102, 109],
         [105, 111, 116],
         ...,
         [119,  88,  87],
         [118,  87,  86],
         [118,  87,  86]],
 
        [[116, 124, 131],
         [107, 115, 122],
         [106, 112, 117],
         ...,
         [122,  91,  90],
         [122,  91,  90],
         [122,  91,  90]],
 
        [[115, 123, 130],
         [114, 122, 129],
         [105, 111, 116],
         ...,
         [122,  91,  90],
         [122,  91,  90],
         [122,  91,  90]],
 
        ...,
 
        [[ 54,  59,  57],
         [ 54,  59,  57],
         [ 54,  59,  57],
         ...,
         [ 86,  82,  81],
         [ 79,  74,  73],
         [ 74,  69,  68]],
 
        [[ 56,  61,  59],
         [ 53,  58,  56],
         [ 49,  54,  52],
         ...,
         [ 86,  82,  81],
         [ 85,  80,  79],
         [ 84,  79,  78]],
 
        [[ 56,  61,  59],
         [ 51,  56,  54],
         [ 45,  50,  48],
         ...,
         [ 91,  87,  86],
         [ 96,  91,  90],
         [100,  95,  94]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2632_jpeg_jpg.rf.f44037edca490b16cbf06427e28ea946.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.065420150756836, 'inference': 12.060165405273438, 'postprocess': 1.5370845794677734},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[166, 165, 151],
         [174, 173, 159],
         [186, 184, 174],
         ...,
         [ 75,  96,  97],
         [ 80, 131, 127],
         [ 95, 163, 156]],
 
        [[164, 159, 150],
         [166, 161, 152],
         [182, 176, 169],
         ...,
         [ 94, 115, 112],
         [103, 154, 147],
         [119, 185, 174]],
 
        [[143, 133, 133],
         [146, 136, 136],
         [170, 157, 159],
         ...,
         [110, 134, 124],
         [129, 177, 165],
         [143, 205, 189]],
 
        ...,
 
        [[ 71,  61,  61],
         [ 72,  62,  62],
         [ 73,  63,  63],
         ...,
         [ 46,  14,   1],
         [ 46,  14,   1],
         [ 47,  15,   2]],
 
        [[ 73,  63,  63],
         [ 73,  63,  63],
         [ 73,  63,  63],
         ...,
         [ 46,  14,   1],
         [ 46,  14,   1],
         [ 47,  15,   2]],
 
        [[ 76,  66,  66],
         [ 75,  65,  65],
         [ 72,  62,  62],
         ...,
         [ 46,  14,   1],
         [ 46,  14,   1],
         [ 47,  15,   2]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_2651_jpeg_jpg.rf.84b3930aa80b610cc97bf1c176763940.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.0585060119628906, 'inference': 11.606693267822266, 'postprocess': 2.3925304412841797},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[144, 169, 209],
         [144, 169, 209],
         [149, 171, 212],
         ...,
         [ 84, 125, 150],
         [ 87, 129, 152],
         [ 88, 132, 155]],
 
        [[134, 159, 199],
         [138, 163, 203],
         [148, 170, 211],
         ...,
         [ 86, 126, 151],
         [ 89, 131, 154],
         [ 90, 134, 157]],
 
        [[135, 160, 200],
         [142, 167, 207],
         [156, 178, 219],
         ...,
         [ 89, 129, 154],
         [ 93, 134, 157],
         [ 95, 137, 160]],
 
        ...,
 
        [[ 23,  18,  19],
         [ 23,  18,  19],
         [ 22,  17,  18],
         ...,
         [ 26,  21,  22],
         [ 25,  20,  21],
         [ 25,  20,  21]],
 
        [[ 24,  19,  20],
         [ 24,  19,  20],
         [ 23,  18,  19],
         ...,
         [ 27,  22,  23],
         [ 26,  21,  22],
         [ 25,  20,  21]],
 
        [[ 25,  20,  21],
         [ 25,  20,  21],
         [ 24,  19,  20],
         ...,
         [ 27,  22,  23],
         [ 26,  21,  22],
         [ 26,  21,  22]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_3129_jpeg_jpg.rf.90c472dcdf9b6713ec767cc97560ceca.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8873214721679688, 'inference': 10.390758514404297, 'postprocess': 1.3701915740966797},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 67,  69,  70],
         [ 67,  69,  70],
         [ 68,  70,  71],
         ...,
         [ 91, 130, 138],
         [ 96, 135, 143],
         [100, 139, 147]],
 
        [[ 70,  72,  73],
         [ 70,  72,  73],
         [ 70,  72,  73],
         ...,
         [ 80, 119, 127],
         [ 83, 122, 130],
         [ 86, 125, 133]],
 
        [[ 72,  74,  75],
         [ 72,  74,  75],
         [ 71,  73,  74],
         ...,
         [ 76, 118, 123],
         [ 77, 119, 124],
         [ 80, 122, 127]],
 
        ...,
 
        [[ 81,  68,  30],
         [ 81,  68,  30],
         [ 82,  69,  31],
         ...,
         [142, 137,  98],
         [145, 140, 101],
         [147, 142, 103]],
 
        [[ 81,  68,  30],
         [ 82,  69,  31],
         [ 83,  70,  32],
         ...,
         [143, 138,  99],
         [146, 141, 102],
         [150, 145, 106]],
 
        [[ 82,  69,  31],
         [ 82,  69,  31],
         [ 84,  71,  33],
         ...,
         [146, 141, 102],
         [150, 145, 106],
         [154, 149, 110]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_3134_jpeg_jpg.rf.50750ca778773042a3c46a1d3e480132.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7516613006591797, 'inference': 11.096477508544922, 'postprocess': 1.638174057006836},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 66,  70,  71],
         [ 68,  72,  73],
         [ 69,  73,  74],
         ...,
         [ 43,  75,  86],
         [ 40,  73,  82],
         [ 64,  97, 106]],
 
        [[ 68,  72,  73],
         [ 69,  73,  74],
         [ 70,  74,  75],
         ...,
         [ 47,  79,  90],
         [ 63,  96, 105],
         [ 93, 126, 135]],
 
        [[ 70,  74,  75],
         [ 71,  75,  76],
         [ 70,  74,  75],
         ...,
         [ 65, 100, 110],
         [ 65,  98, 107],
         [ 63,  96, 105]],
 
        ...,
 
        [[ 18,  10,   3],
         [ 18,  10,   3],
         [ 18,  10,   3],
         ...,
         [173, 180, 143],
         [179, 186, 149],
         [184, 191, 154]],
 
        [[ 18,  10,   3],
         [ 18,  10,   3],
         [ 18,  10,   3],
         ...,
         [179, 186, 149],
         [181, 188, 151],
         [182, 189, 152]],
 
        [[ 18,  10,   3],
         [ 18,  10,   3],
         [ 18,  10,   3],
         ...,
         [181, 188, 151],
         [180, 187, 150],
         [177, 184, 147]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_3136_jpeg_jpg.rf.0d8fef73d4cc5e1c35ce424444d9e44b.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7800331115722656, 'inference': 12.836694717407227, 'postprocess': 2.5544166564941406},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[111, 131, 132],
         [107, 127, 128],
         [109, 127, 128],
         ...,
         [125, 132, 152],
         [130, 135, 156],
         [104, 109, 130]],
 
        [[ 82, 100, 101],
         [ 78,  96,  97],
         [ 78,  96,  97],
         ...,
         [112, 119, 139],
         [125, 130, 151],
         [115, 120, 141]],
 
        [[ 65,  80,  83],
         [ 60,  75,  78],
         [ 59,  74,  77],
         ...,
         [ 99, 106, 126],
         [115, 122, 141],
         [126, 133, 152]],
 
        ...,
 
        [[ 66,  55,  25],
         [ 65,  54,  24],
         [ 65,  54,  24],
         ...,
         [ 43,  27,  10],
         [ 44,  28,  11],
         [ 44,  28,  11]],
 
        [[ 68,  57,  27],
         [ 67,  56,  26],
         [ 65,  54,  24],
         ...,
         [ 42,  26,   9],
         [ 43,  27,  10],
         [ 43,  27,  10]],
 
        [[ 69,  58,  28],
         [ 68,  57,  27],
         [ 66,  55,  25],
         ...,
         [ 41,  25,   8],
         [ 41,  25,   8],
         [ 42,  26,   9]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_3144_jpeg_jpg.rf.f29a36360174dc83ecef93275ed8f02e.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7056465148925781, 'inference': 13.962507247924805, 'postprocess': 1.1904239654541016},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 93,  71,  53],
         [ 93,  71,  53],
         [ 95,  73,  55],
         ...,
         [111, 103,  86],
         [101,  93,  76],
         [ 82,  74,  57]],
 
        [[ 93,  71,  53],
         [ 94,  72,  54],
         [ 95,  73,  55],
         ...,
         [109, 101,  84],
         [ 97,  89,  72],
         [ 79,  71,  54]],
 
        [[ 94,  72,  54],
         [ 95,  73,  55],
         [ 96,  74,  56],
         ...,
         [110, 100,  83],
         [ 96,  86,  69],
         [ 79,  69,  52]],
 
        ...,
 
        [[ 27,  15,   3],
         [ 29,  17,   5],
         [ 30,  18,   6],
         ...,
         [165, 156, 142],
         [176, 160, 147],
         [183, 168, 152]],
 
        [[ 27,  15,   3],
         [ 29,  17,   5],
         [ 30,  18,   6],
         ...,
         [172, 163, 153],
         [178, 164, 152],
         [180, 167, 153]],
 
        [[ 27,  15,   3],
         [ 29,  17,   5],
         [ 30,  18,   6],
         ...,
         [183, 174, 164],
         [184, 172, 162],
         [186, 172, 160]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_3154_jpeg_jpg.rf.5f429a366c02d38bc9e2217f4508c3e0.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8315315246582031, 'inference': 10.975122451782227, 'postprocess': 2.281665802001953},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[110, 141, 164],
         [ 94, 125, 148],
         [ 98, 129, 152],
         ...,
         [172, 130, 118],
         [173, 131, 119],
         [174, 132, 120]],
 
        [[112, 143, 166],
         [103, 134, 157],
         [102, 133, 156],
         ...,
         [174, 132, 120],
         [175, 133, 121],
         [176, 134, 122]],
 
        [[109, 140, 163],
         [109, 140, 163],
         [101, 132, 155],
         ...,
         [178, 136, 124],
         [179, 137, 125],
         [180, 138, 126]],
 
        ...,
 
        [[ 70,  84,  60],
         [ 70,  84,  60],
         [ 74,  88,  64],
         ...,
         [ 53,  49,  25],
         [ 52,  48,  24],
         [ 52,  48,  24]],
 
        [[ 72,  86,  62],
         [ 69,  83,  59],
         [ 69,  83,  59],
         ...,
         [ 54,  50,  26],
         [ 53,  49,  25],
         [ 53,  49,  25]],
 
        [[ 75,  89,  65],
         [ 68,  82,  58],
         [ 66,  80,  56],
         ...,
         [ 54,  50,  26],
         [ 54,  50,  26],
         [ 54,  50,  26]]], dtype=uint8)
 orig_shape: (1024, 768)
 path: '/content/Aquarium_Data/test/images/IMG_3164_jpeg_jpg.rf.06637eee0b72df791aa729807ca45c4d.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.9066333770751953, 'inference': 12.214183807373047, 'postprocess': 2.3369789123535156},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 97,  78,  70],
         [ 97,  78,  70],
         [ 96,  77,  69],
         ...,
         [149,  99,  87],
         [149,  99,  87],
         [150, 100,  88]],
 
        [[ 99,  80,  72],
         [ 99,  80,  72],
         [ 98,  79,  71],
         ...,
         [147,  97,  85],
         [147,  97,  85],
         [147,  97,  85]],
 
        [[102,  83,  75],
         [101,  82,  74],
         [101,  82,  74],
         ...,
         [144,  95,  85],
         [144,  95,  85],
         [143,  94,  84]],
 
        ...,
 
        [[101, 114,  82],
         [ 97, 110,  78],
         [ 94, 107,  75],
         ...,
         [ 36,  28,   5],
         [ 35,  27,   4],
         [ 34,  26,   3]],
 
        [[ 92, 105,  73],
         [ 92, 105,  73],
         [ 92, 105,  73],
         ...,
         [ 35,  27,   4],
         [ 33,  25,   2],
         [ 32,  24,   1]],
 
        [[ 87, 100,  68],
         [ 89, 102,  70],
         [ 92, 105,  73],
         ...,
         [ 34,  26,   3],
         [ 32,  24,   1],
         [ 31,  23,   0]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_3173_jpeg_jpg.rf.6f05acaa0b22d410a5df3ea3286e227d.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7483234405517578, 'inference': 11.801004409790039, 'postprocess': 2.202272415161133},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[116, 102,  90],
         [116, 102,  90],
         [115, 101,  89],
         ...,
         [165, 124, 115],
         [164, 123, 114],
         [163, 122, 113]],
 
        [[111,  97,  85],
         [111,  97,  85],
         [111,  97,  85],
         ...,
         [163, 122, 113],
         [163, 122, 113],
         [163, 122, 113]],
 
        [[109,  93,  81],
         [109,  93,  81],
         [110,  94,  82],
         ...,
         [159, 120, 111],
         [160, 121, 112],
         [160, 121, 112]],
 
        ...,
 
        [[135, 157, 115],
         [137, 159, 117],
         [140, 162, 120],
         ...,
         [ 63,  57,  22],
         [ 62,  56,  21],
         [ 60,  54,  19]],
 
        [[133, 155, 113],
         [136, 158, 116],
         [139, 161, 119],
         ...,
         [ 64,  58,  23],
         [ 62,  56,  21],
         [ 60,  54,  19]],
 
        [[133, 155, 113],
         [136, 158, 116],
         [139, 161, 119],
         ...,
         [ 64,  58,  23],
         [ 62,  56,  21],
         [ 60,  54,  19]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_3175_jpeg_jpg.rf.686c7d36e049eea974a363e99bf0bee0.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7557144165039062, 'inference': 10.970592498779297, 'postprocess': 2.6764869689941406},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 86, 111, 107],
         [ 98, 123, 119],
         [105, 130, 126],
         ...,
         [181, 187, 194],
         [187, 193, 200],
         [197, 203, 210]],
 
        [[ 87, 111, 109],
         [ 96, 121, 117],
         [105, 130, 126],
         ...,
         [178, 184, 191],
         [185, 191, 198],
         [195, 201, 208]],
 
        [[ 86, 112, 112],
         [ 93, 120, 117],
         [105, 132, 129],
         ...,
         [180, 186, 193],
         [183, 189, 196],
         [189, 195, 202]],
 
        ...,
 
        [[ 83,  88,  86],
         [ 85,  90,  88],
         [ 87,  92,  90],
         ...,
         [162, 158, 164],
         [157, 150, 157],
         [163, 156, 163]],
 
        [[ 78,  83,  81],
         [ 80,  85,  83],
         [ 83,  88,  86],
         ...,
         [151, 147, 153],
         [143, 136, 143],
         [148, 141, 148]],
 
        [[ 72,  77,  75],
         [ 75,  80,  78],
         [ 79,  84,  82],
         ...,
         [151, 147, 153],
         [154, 147, 154],
         [167, 160, 167]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_8331_jpg.rf.ec024bdf1e9de02b020b5e6505c1c58b.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8591880798339844, 'inference': 10.65826416015625, 'postprocess': 2.2819042205810547},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 44,  40,  16],
         [ 42,  38,  14],
         [ 41,  37,  13],
         ...,
         [ 50,  71,  63],
         [ 50,  73,  65],
         [ 51,  74,  66]],
 
        [[ 46,  42,  18],
         [ 44,  40,  16],
         [ 43,  39,  15],
         ...,
         [ 50,  71,  63],
         [ 49,  72,  64],
         [ 50,  73,  65]],
 
        [[ 48,  44,  20],
         [ 46,  42,  18],
         [ 45,  41,  17],
         ...,
         [ 49,  70,  62],
         [ 49,  72,  64],
         [ 49,  72,  64]],
 
        ...,
 
        [[ 54,  49,  24],
         [ 50,  45,  20],
         [ 46,  41,  16],
         ...,
         [ 61,  93, 104],
         [ 61,  93, 104],
         [ 61,  93, 104]],
 
        [[ 55,  50,  25],
         [ 50,  45,  20],
         [ 45,  40,  15],
         ...,
         [ 64,  96, 107],
         [ 62,  94, 105],
         [ 61,  93, 104]],
 
        [[ 54,  49,  24],
         [ 49,  44,  19],
         [ 44,  39,  14],
         ...,
         [ 66,  98, 109],
         [ 63,  95, 106],
         [ 62,  94, 105]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_8343_jpg.rf.2d88000497d74d72aedc118b125a0c07.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.9087791442871094, 'inference': 11.235237121582031, 'postprocess': 2.354145050048828},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 47,  67,  68],
         [ 49,  69,  70],
         [ 52,  72,  73],
         ...,
         [  3,   4,   2],
         [  3,   4,   2],
         [  3,   4,   2]],
 
        [[ 47,  67,  68],
         [ 50,  70,  71],
         [ 52,  72,  73],
         ...,
         [  3,   4,   2],
         [  3,   4,   2],
         [  3,   4,   2]],
 
        [[ 47,  67,  68],
         [ 49,  69,  70],
         [ 52,  72,  73],
         ...,
         [  3,   4,   2],
         [  3,   4,   2],
         [  3,   4,   2]],
 
        ...,
 
        [[112, 131,  88],
         [123, 142,  99],
         [139, 158, 115],
         ...,
         [ 34,  46,  50],
         [ 30,  42,  46],
         [ 30,  42,  46]],
 
        [[115, 134,  91],
         [122, 141,  98],
         [138, 157, 114],
         ...,
         [ 37,  49,  53],
         [ 32,  44,  48],
         [ 29,  41,  45]],
 
        [[107, 126,  83],
         [110, 129,  86],
         [125, 144, 101],
         ...,
         [ 39,  51,  55],
         [ 32,  44,  48],
         [ 29,  41,  45]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_8395_jpg.rf.3bebece033961c9f665571644a14261f.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8422603607177734, 'inference': 10.58506965637207, 'postprocess': 2.397775650024414},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 64,  82,  81],
         [ 64,  82,  81],
         [ 63,  81,  80],
         ...,
         [  5,   8,   6],
         [  5,   8,   6],
         [  5,   8,   6]],
 
        [[ 66,  84,  83],
         [ 66,  84,  83],
         [ 64,  82,  81],
         ...,
         [  5,   8,   6],
         [  4,   7,   5],
         [  3,   6,   4]],
 
        [[ 64,  82,  81],
         [ 64,  82,  81],
         [ 63,  81,  80],
         ...,
         [  6,   9,   7],
         [  3,   6,   4],
         [  1,   4,   2]],
 
        ...,
 
        [[135, 155, 108],
         [144, 164, 117],
         [149, 169, 122],
         ...,
         [  9,  23,  21],
         [ 14,  28,  26],
         [ 28,  42,  40]],
 
        [[145, 165, 118],
         [154, 174, 127],
         [155, 175, 128],
         ...,
         [  9,  23,  21],
         [  9,  23,  21],
         [ 22,  36,  34]],
 
        [[156, 176, 129],
         [162, 182, 135],
         [159, 179, 132],
         ...,
         [ 25,  39,  37],
         [ 25,  39,  37],
         [ 39,  53,  51]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_8396_jpg.rf.106a6ced5c649ea81f0de8ecaa4ff3b8.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.7490386962890625, 'inference': 10.832786560058594, 'postprocess': 2.390623092651367},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[104, 108, 109],
         [100, 104, 105],
         [101, 105, 106],
         ...,
         [187, 198, 136],
         [188, 199, 137],
         [188, 199, 137]],
 
        [[107, 111, 112],
         [103, 107, 108],
         [103, 107, 108],
         ...,
         [185, 196, 134],
         [187, 198, 136],
         [187, 198, 136]],
 
        [[107, 111, 112],
         [104, 108, 109],
         [104, 108, 109],
         ...,
         [183, 193, 133],
         [184, 194, 134],
         [184, 194, 134]],
 
        ...,
 
        [[ 24,  16,   0],
         [ 28,  20,   3],
         [ 35,  27,  10],
         ...,
         [209, 218, 155],
         [210, 219, 156],
         [211, 220, 157]],
 
        [[ 25,  17,   0],
         [ 31,  23,   6],
         [ 38,  30,  13],
         ...,
         [208, 217, 154],
         [209, 218, 155],
         [209, 218, 155]],
 
        [[ 28,  20,   3],
         [ 34,  26,   9],
         [ 41,  33,  16],
         ...,
         [208, 217, 154],
         [210, 219, 156],
         [210, 219, 156]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_8404_jpg.rf.265b89e862a375f6b89f781ea60ed480.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.6531944274902344, 'inference': 10.650873184204102, 'postprocess': 2.4809837341308594},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 63,  50,  48],
         [ 61,  49,  45],
         [ 60,  48,  42],
         ...,
         [179, 202, 148],
         [174, 197, 143],
         [171, 194, 140]],
 
        [[ 57,  44,  42],
         [ 57,  45,  41],
         [ 56,  44,  38],
         ...,
         [170, 193, 139],
         [167, 190, 136],
         [165, 188, 134]],
 
        [[ 49,  37,  33],
         [ 55,  43,  39],
         [ 56,  44,  38],
         ...,
         [168, 191, 137],
         [169, 192, 138],
         [170, 193, 139]],
 
        ...,
 
        [[  3,   4,   2],
         [  3,   4,   2],
         [  1,   4,   2],
         ...,
         [ 60,  69,  43],
         [ 60,  69,  43],
         [ 56,  65,  39]],
 
        [[  6,   6,   6],
         [  5,   5,   5],
         [  1,   3,   3],
         ...,
         [ 56,  65,  39],
         [ 55,  64,  38],
         [ 51,  60,  34]],
 
        [[  9,   9,   9],
         [  7,   7,   7],
         [  2,   4,   4],
         ...,
         [ 58,  67,  41],
         [ 61,  70,  44],
         [ 62,  71,  45]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_8420_jpg.rf.31f1d5f1440e48ccf1dee988b565911b.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.8153190612792969, 'inference': 10.989189147949219, 'postprocess': 2.3546218872070312},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[106, 119, 121],
         [106, 119, 121],
         [107, 120, 122],
         ...,
         [ 80,  79,  75],
         [ 74,  73,  69],
         [ 67,  66,  62]],
 
        [[103, 116, 118],
         [104, 117, 119],
         [105, 118, 120],
         ...,
         [ 82,  81,  77],
         [ 76,  75,  71],
         [ 70,  69,  65]],
 
        [[100, 113, 115],
         [101, 114, 116],
         [101, 114, 116],
         ...,
         [ 82,  81,  77],
         [ 75,  74,  70],
         [ 69,  68,  64]],
 
        ...,
 
        [[ 12,  16,  11],
         [ 11,  15,  10],
         [  9,  13,   8],
         ...,
         [ 89,  80, 100],
         [ 81,  72,  92],
         [ 88,  79,  99]],
 
        [[  7,  11,   6],
         [  9,  13,   8],
         [ 10,  14,   9],
         ...,
         [ 98,  87, 107],
         [ 90,  79,  99],
         [ 98,  87, 107]],
 
        [[  2,   6,   1],
         [  7,  11,   6],
         [ 11,  15,  10],
         ...,
         [113, 102, 122],
         [ 98,  87, 107],
         [ 99,  88, 108]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_8452_jpg.rf.6bbff701ab93e29553b3a70137fd4e66.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.9347667694091797, 'inference': 11.561155319213867, 'postprocess': 1.5206336975097656},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[28, 26, 25],
         [28, 26, 25],
         [18, 16, 15],
         ...,
         [ 2,  5, 10],
         [10, 13, 18],
         [21, 24, 29]],
 
        [[23, 21, 20],
         [24, 22, 21],
         [14, 12, 11],
         ...,
         [ 7, 10, 15],
         [12, 15, 20],
         [20, 23, 28]],
 
        [[18, 18, 18],
         [20, 20, 20],
         [ 9,  9,  9],
         ...,
         [12, 15, 20],
         [13, 16, 21],
         [17, 20, 25]],
 
        ...,
 
        [[23, 12, 14],
         [27, 16, 18],
         [28, 17, 19],
         ...,
         [74, 68, 79],
         [74, 68, 79],
         [73, 67, 78]],
 
        [[25, 14, 16],
         [29, 18, 20],
         [30, 19, 21],
         ...,
         [70, 64, 75],
         [69, 63, 74],
         [68, 62, 73]],
 
        [[37, 26, 28],
         [40, 29, 31],
         [39, 28, 30],
         ...,
         [69, 63, 74],
         [68, 62, 73],
         [67, 61, 72]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_8490_jpg.rf.1836542cf054c6d303a2dd05d4194d7f.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.9118785858154297, 'inference': 11.356830596923828, 'postprocess': 2.566099166870117},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[17, 85, 62],
         [ 6, 74, 51],
         [12, 80, 57],
         ...,
         [75, 76, 44],
         [77, 76, 42],
         [80, 79, 45]],
 
        [[ 1, 69, 46],
         [ 0, 65, 42],
         [ 8, 76, 53],
         ...,
         [71, 72, 40],
         [73, 72, 38],
         [77, 76, 42]],
 
        [[ 0, 66, 43],
         [ 0, 67, 44],
         [13, 81, 58],
         ...,
         [73, 71, 40],
         [73, 71, 40],
         [76, 74, 43]],
 
        ...,
 
        [[15, 20, 18],
         [16, 21, 19],
         [13, 18, 16],
         ...,
         [ 8, 15, 12],
         [ 6, 13, 10],
         [ 4, 11,  8]],
 
        [[13, 18, 16],
         [14, 19, 17],
         [12, 17, 15],
         ...,
         [ 8, 15, 12],
         [ 6, 13, 10],
         [ 3, 10,  7]],
 
        [[ 9, 14, 12],
         [10, 15, 13],
         [ 8, 13, 11],
         ...,
         [10, 17, 14],
         [ 9, 16, 13],
         [ 7, 14, 11]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8497_MOV-0_jpg.rf.5c59bd1bf7d8fd7a20999d51a79a12c0.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 2.0024776458740234, 'inference': 16.483068466186523, 'postprocess': 2.6824474334716797},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[92, 81, 59],
         [91, 80, 58],
         [91, 80, 58],
         ...,
         [56, 63, 58],
         [60, 68, 61],
         [64, 72, 65]],
 
        [[91, 80, 58],
         [91, 80, 58],
         [91, 80, 58],
         ...,
         [54, 61, 56],
         [55, 63, 56],
         [58, 66, 59]],
 
        [[89, 78, 56],
         [89, 78, 56],
         [89, 78, 56],
         ...,
         [55, 62, 57],
         [56, 64, 57],
         [58, 66, 59]],
 
        ...,
 
        [[19, 25, 24],
         [16, 22, 21],
         [15, 21, 20],
         ...,
         [ 0,  2,  2],
         [ 1,  3,  3],
         [ 2,  4,  4]],
 
        [[15, 21, 20],
         [16, 22, 21],
         [16, 22, 21],
         ...,
         [ 0,  2,  2],
         [ 1,  3,  3],
         [ 2,  4,  4]],
 
        [[13, 19, 18],
         [15, 21, 20],
         [18, 24, 23],
         ...,
         [ 0,  2,  2],
         [ 1,  3,  3],
         [ 2,  4,  4]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8497_MOV-3_jpg.rf.fd813e14681c8b41e709a500748ce46a.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.5418529510498047, 'inference': 11.681318283081055, 'postprocess': 2.466917037963867},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 86, 204, 156],
         [ 79, 199, 151],
         [ 78, 200, 152],
         ...,
         [ 87,  85,  50],
         [ 85,  79,  44],
         [ 87,  81,  46]],
 
        [[ 67, 183, 136],
         [ 64, 182, 134],
         [ 68, 188, 140],
         ...,
         [ 93,  91,  56],
         [ 90,  84,  49],
         [ 91,  85,  50]],
 
        [[ 53, 168, 119],
         [ 52, 167, 118],
         [ 56, 173, 124],
         ...,
         [ 94,  92,  57],
         [ 91,  85,  50],
         [ 91,  85,  50]],
 
        ...,
 
        [[ 60, 184,  90],
         [ 50, 172,  78],
         [ 37, 155,  66],
         ...,
         [  3,   9,   4],
         [  4,  10,   5],
         [  4,  10,   5]],
 
        [[ 30, 150,  56],
         [ 19, 136,  43],
         [ 10, 122,  34],
         ...,
         [  3,   9,   4],
         [  4,  10,   5],
         [  4,  10,   5]],
 
        [[ 15, 132,  39],
         [  3, 118,  25],
         [  0, 106,  18],
         ...,
         [  3,   9,   4],
         [  4,  10,   5],
         [  4,  10,   5]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8497_MOV-5_jpg.rf.3deffb208d656b7845661c5e33dd1afb.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.5559196472167969, 'inference': 11.064291000366211, 'postprocess': 2.4611949920654297},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 5,  5,  5],
         [ 5,  5,  5],
         [ 5,  5,  5],
         ...,
         [ 3,  7,  2],
         [ 7, 11,  6],
         [10, 14,  9]],
 
        [[ 5,  5,  5],
         [ 5,  5,  5],
         [ 5,  5,  5],
         ...,
         [ 4,  8,  3],
         [ 5,  9,  4],
         [ 6, 10,  5]],
 
        [[ 4,  4,  4],
         [ 5,  5,  5],
         [ 5,  5,  5],
         ...,
         [ 4,  8,  3],
         [ 3,  7,  2],
         [ 3,  7,  2]],
 
        ...,
 
        [[50, 58, 41],
         [52, 60, 43],
         [55, 63, 46],
         ...,
         [27, 18, 14],
         [28, 19, 15],
         [29, 20, 16]],
 
        [[57, 67, 50],
         [59, 69, 52],
         [60, 70, 53],
         ...,
         [26, 17, 13],
         [29, 20, 16],
         [32, 23, 19]],
 
        [[64, 74, 57],
         [66, 76, 59],
         [66, 76, 59],
         ...,
         [23, 14, 10],
         [28, 19, 15],
         [33, 24, 20]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8513_MOV-0_jpg.rf.2a2f77e3f73630b60aaf6ad3ca4ed130.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.6140937805175781, 'inference': 11.25478744506836, 'postprocess': 2.311229705810547},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[11, 12, 10],
         [11, 12, 10],
         [10, 11,  9],
         ...,
         [ 2,  3,  1],
         [ 2,  3,  1],
         [ 2,  3,  1]],
 
        [[ 5,  6,  4],
         [ 5,  6,  4],
         [ 5,  6,  4],
         ...,
         [ 2,  3,  1],
         [ 2,  3,  1],
         [ 2,  3,  1]],
 
        [[ 7,  8,  6],
         [ 8,  9,  7],
         [ 8,  9,  7],
         ...,
         [ 3,  4,  2],
         [ 2,  3,  1],
         [ 2,  3,  1]],
 
        ...,
 
        [[16, 18, 18],
         [16, 18, 18],
         [16, 18, 18],
         ...,
         [ 2,  2,  2],
         [ 1,  1,  1],
         [ 0,  0,  0]],
 
        [[21, 23, 23],
         [19, 21, 21],
         [17, 19, 19],
         ...,
         [ 1,  1,  1],
         [ 1,  1,  1],
         [ 1,  1,  1]],
 
        [[19, 21, 21],
         [20, 22, 22],
         [21, 23, 23],
         ...,
         [ 0,  0,  0],
         [ 1,  1,  1],
         [ 2,  2,  2]]], dtype=uint8)
 orig_shape: (768, 1024)
 path: '/content/Aquarium_Data/test/images/IMG_8515_jpg.rf.98a9daca7c5a5bad9872bd7fb2d4f198.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.850128173828125, 'inference': 14.51420783996582, 'postprocess': 2.8617382049560547},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[57, 39,  2],
         [59, 41,  4],
         [58, 40,  3],
         ...,
         [88, 72,  5],
         [89, 73,  6],
         [89, 73,  6]],
 
        [[61, 43,  6],
         [62, 44,  7],
         [60, 42,  5],
         ...,
         [88, 72,  5],
         [88, 72,  5],
         [88, 72,  5]],
 
        [[62, 44,  7],
         [63, 45,  8],
         [61, 43,  6],
         ...,
         [87, 71,  4],
         [87, 71,  4],
         [87, 71,  4]],
 
        ...,
 
        [[43, 42, 38],
         [49, 48, 44],
         [49, 48, 44],
         ...,
         [57, 55, 47],
         [55, 53, 45],
         [52, 50, 42]],
 
        [[42, 41, 37],
         [47, 46, 42],
         [47, 46, 42],
         ...,
         [52, 49, 41],
         [50, 47, 39],
         [48, 45, 37]],
 
        [[36, 35, 31],
         [40, 39, 35],
         [40, 39, 35],
         ...,
         [43, 40, 32],
         [39, 36, 28],
         [37, 34, 26]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8582_MOV-0_jpg.rf.aa8304d7a5112d63c8841d96160d42cd.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.645803451538086, 'inference': 11.895895004272461, 'postprocess': 2.272367477416992},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 53,  51,  27],
         [ 53,  51,  27],
         [ 54,  52,  28],
         ...,
         [ 57,  50,  25],
         [ 58,  51,  26],
         [ 60,  53,  28]],
 
        [[ 52,  50,  26],
         [ 53,  51,  27],
         [ 54,  52,  28],
         ...,
         [ 55,  48,  23],
         [ 54,  47,  22],
         [ 54,  47,  22]],
 
        [[ 52,  50,  26],
         [ 53,  51,  27],
         [ 53,  51,  27],
         ...,
         [ 55,  48,  23],
         [ 53,  46,  21],
         [ 52,  45,  20]],
 
        ...,
 
        [[215, 246, 207],
         [218, 247, 208],
         [220, 247, 207],
         ...,
         [ 60,  71,  85],
         [ 60,  71,  85],
         [ 60,  71,  85]],
 
        [[217, 247, 206],
         [218, 248, 207],
         [220, 247, 207],
         ...,
         [ 58,  69,  83],
         [ 56,  67,  81],
         [ 55,  66,  80]],
 
        [[217, 247, 206],
         [218, 248, 207],
         [220, 247, 207],
         ...,
         [ 58,  69,  83],
         [ 55,  66,  80],
         [ 53,  64,  78]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8582_MOV-3_jpg.rf.c7dde0639837077f76428d70223368a4.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.5192031860351562, 'inference': 10.710000991821289, 'postprocess': 1.3790130615234375},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[ 79,  75,  34],
         [ 84,  80,  39],
         [ 83,  79,  38],
         ...,
         [134, 117,  24],
         [135, 118,  25],
         [136, 119,  26]],
 
        [[ 79,  75,  34],
         [ 84,  80,  39],
         [ 83,  79,  38],
         ...,
         [133, 116,  23],
         [133, 116,  23],
         [133, 116,  23]],
 
        [[ 80,  76,  35],
         [ 84,  80,  39],
         [ 84,  80,  39],
         ...,
         [134, 116,  25],
         [134, 116,  25],
         [134, 116,  25]],
 
        ...,
 
        [[ 74,  94,  81],
         [ 77,  97,  84],
         [ 75,  92,  83],
         ...,
         [ 69,  68,  70],
         [ 73,  72,  74],
         [ 75,  74,  76]],
 
        [[ 60,  80,  67],
         [ 70,  90,  77],
         [ 71,  88,  79],
         ...,
         [ 63,  62,  64],
         [ 68,  67,  69],
         [ 72,  71,  73]],
 
        [[ 45,  65,  52],
         [ 60,  80,  67],
         [ 62,  79,  70],
         ...,
         [ 60,  59,  61],
         [ 66,  65,  67],
         [ 71,  70,  72]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8582_MOV-5_jpg.rf.9d7a26fbf145ce39ab0831b4e6bc1f1e.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.4803409576416016, 'inference': 10.815620422363281, 'postprocess': 1.3570785522460938},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[226,  91,   0],
         [226,  91,   0],
         [226,  91,   0],
         ...,
         [251, 104,   0],
         [251, 104,   0],
         [251, 104,   0]],
 
        [[226,  91,   0],
         [226,  91,   0],
         [226,  91,   0],
         ...,
         [251, 104,   0],
         [251, 104,   0],
         [251, 104,   0]],
 
        [[226,  91,   0],
         [226,  91,   0],
         [226,  91,   0],
         ...,
         [251, 104,   0],
         [251, 104,   0],
         [251, 104,   0]],
 
        ...,
 
        [[206,  81,   1],
         [207,  82,   2],
         [207,  82,   2],
         ...,
         [248,  97,   0],
         [248,  97,   0],
         [248,  97,   0]],
 
        [[206,  81,   1],
         [206,  81,   1],
         [207,  82,   2],
         ...,
         [248,  97,   0],
         [248,  97,   0],
         [248,  97,   0]],
 
        [[206,  81,   1],
         [206,  81,   1],
         [206,  81,   1],
         ...,
         [248,  97,   0],
         [248,  97,   0],
         [248,  97,   0]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8590_MOV-2_jpg.rf.2136fdb5dcbcd58a1dc456bb3e5bf476.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.4073848724365234, 'inference': 9.969711303710938, 'postprocess': 2.3772716522216797},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[253, 101,   0],
         [253, 101,   0],
         [253, 101,   0],
         ...,
         [255, 107,   1],
         [255, 107,   1],
         [255, 107,   1]],
 
        [[253, 101,   0],
         [253, 101,   0],
         [253, 101,   0],
         ...,
         [255, 107,   1],
         [255, 107,   1],
         [255, 107,   1]],
 
        [[253, 101,   0],
         [253, 101,   0],
         [253, 101,   0],
         ...,
         [255, 107,   1],
         [255, 107,   1],
         [255, 107,   1]],
 
        ...,
 
        [[104,  27,   0],
         [104,  27,   0],
         [104,  27,   0],
         ...,
         [ 94,  28,   0],
         [ 94,  28,   0],
         [ 94,  28,   0]],
 
        [[104,  27,   0],
         [104,  27,   0],
         [104,  27,   0],
         ...,
         [ 94,  28,   0],
         [ 94,  28,   0],
         [ 94,  28,   0]],
 
        [[104,  27,   0],
         [104,  27,   0],
         [104,  27,   0],
         ...,
         [ 94,  28,   0],
         [ 94,  28,   0],
         [ 94,  28,   0]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8590_MOV-5_jpg.rf.074e6d8acdd3fcad16d866c341b43769.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.415252685546875, 'inference': 10.025262832641602, 'postprocess': 2.1643638610839844},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[24, 10,  4],
         [27, 13,  7],
         [31, 17, 11],
         ...,
         [38, 24, 12],
         [39, 25, 13],
         [40, 26, 14]],
 
        [[25, 11,  5],
         [28, 14,  8],
         [31, 17, 11],
         ...,
         [37, 23, 11],
         [37, 23, 11],
         [37, 23, 11]],
 
        [[27, 13,  7],
         [29, 15,  9],
         [32, 18, 12],
         ...,
         [36, 22, 10],
         [37, 23, 11],
         [37, 23, 11]],
 
        ...,
 
        [[70, 77, 64],
         [72, 79, 66],
         [73, 80, 65],
         ...,
         [76, 79, 57],
         [74, 77, 55],
         [72, 75, 53]],
 
        [[72, 79, 66],
         [72, 79, 66],
         [72, 79, 64],
         ...,
         [74, 77, 55],
         [74, 77, 55],
         [73, 76, 54]],
 
        [[73, 80, 67],
         [71, 78, 65],
         [71, 78, 63],
         ...,
         [74, 77, 55],
         [76, 79, 57],
         [77, 80, 58]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8595_MOV-0_jpg.rf.312ab0b8b9fca18134aee88044f45a06.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.4121532440185547, 'inference': 10.019779205322266, 'postprocess': 2.310514450073242},
 ultralytics.engine.results.Results object with attributes:
 
 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'fish', 1: 'jellyfish', 2: 'penguin', 3: 'puffin', 4: 'shark', 5: 'starfish', 6: 'stingray'}
 orig_img: array([[[142,  63,   0],
         [142,  63,   0],
         [142,  63,   0],
         ...,
         [119,  55,   0],
         [119,  55,   0],
         [119,  55,   0]],
 
        [[142,  63,   0],
         [142,  63,   0],
         [142,  63,   0],
         ...,
         [119,  55,   0],
         [119,  55,   0],
         [119,  55,   0]],
 
        [[142,  63,   0],
         [142,  63,   0],
         [142,  63,   0],
         ...,
         [119,  55,   0],
         [119,  55,   0],
         [119,  55,   0]],
 
        ...,
 
        [[197, 101,   1],
         [197, 101,   1],
         [197, 101,   1],
         ...,
         [185,  89,   0],
         [185,  89,   0],
         [185,  89,   0]],
 
        [[197, 101,   1],
         [197, 101,   1],
         [197, 101,   1],
         ...,
         [185,  89,   0],
         [185,  89,   0],
         [185,  89,   0]],
 
        [[197, 101,   1],
         [197, 101,   1],
         [197, 101,   1],
         ...,
         [185,  89,   0],
         [185,  89,   0],
         [185,  89,   0]]], dtype=uint8)
 orig_shape: (1024, 576)
 path: '/content/Aquarium_Data/test/images/IMG_8599_MOV-3_jpg.rf.412ebb16ea80e964b4464c50e757df0e.jpg'
 probs: None
 save_dir: 'runs/detect/train34'
 speed: {'preprocess': 1.4886856079101562, 'inference': 10.084390640258789, 'postprocess': 2.4328231811523438}]
[ ]
  1
import numpy as np
[ ]
123456789
for result in results:

    uniq, cnt = np.unique(result.boxes.cls.cpu().numpy(), return_counts=True)  # Torch.Tensor -> numpy
    uniq_cnt_dict = dict(zip(uniq, cnt))

    print('\n{class num:counts} =', uniq_cnt_dict,'\n')

    for c in result.boxes.cls:
        print('class num =', int(c), ', class_name =', model.names[int(c)])
output

{class num:counts} = {4.0: 1, 6.0: 1} 

class num = 4 , class_name = shark
class num = 6 , class_name = stingray

{class num:counts} = {2.0: 4} 

class num = 2 , class_name = penguin
class num = 2 , class_name = penguin
class num = 2 , class_name = penguin
class num = 2 , class_name = penguin

{class num:counts} = {2.0: 4} 

class num = 2 , class_name = penguin
class num = 2 , class_name = penguin
class num = 2 , class_name = penguin
class num = 2 , class_name = penguin

{class num:counts} = {2.0: 1} 

class num = 2 , class_name = penguin

{class num:counts} = {} 


{class num:counts} = {0.0: 3} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 1} 

class num = 0 , class_name = fish

{class num:counts} = {0.0: 2} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {5.0: 1} 

class num = 5 , class_name = starfish

{class num:counts} = {0.0: 1} 

class num = 0 , class_name = fish

{class num:counts} = {0.0: 11, 4.0: 1} 

class num = 4 , class_name = shark
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 3} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 2, 4.0: 2} 

class num = 4 , class_name = shark
class num = 4 , class_name = shark
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 4, 4.0: 2} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 4 , class_name = shark
class num = 4 , class_name = shark
class num = 0 , class_name = fish

{class num:counts} = {0.0: 12, 4.0: 1} 

class num = 4 , class_name = shark
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {1.0: 15} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {1.0: 2} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {1.0: 6} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {1.0: 15} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {1.0: 2} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {1.0: 6} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {0.0: 2, 4.0: 2} 

class num = 4 , class_name = shark
class num = 0 , class_name = fish
class num = 4 , class_name = shark
class num = 0 , class_name = fish

{class num:counts} = {2.0: 1, 4.0: 1} 

class num = 4 , class_name = shark
class num = 2 , class_name = penguin

{class num:counts} = {4.0: 2} 

class num = 4 , class_name = shark
class num = 4 , class_name = shark

{class num:counts} = {} 


{class num:counts} = {4.0: 1} 

class num = 4 , class_name = shark

{class num:counts} = {5.0: 2} 

class num = 5 , class_name = starfish
class num = 5 , class_name = starfish

{class num:counts} = {} 


{class num:counts} = {0.0: 7, 4.0: 2} 

class num = 4 , class_name = shark
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 4 , class_name = shark
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {1.0: 9} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {4.0: 2} 

class num = 4 , class_name = shark
class num = 4 , class_name = shark

{class num:counts} = {6.0: 3} 

class num = 6 , class_name = stingray
class num = 6 , class_name = stingray
class num = 6 , class_name = stingray

{class num:counts} = {4.0: 1} 

class num = 4 , class_name = shark

{class num:counts} = {} 


{class num:counts} = {6.0: 1} 

class num = 6 , class_name = stingray

{class num:counts} = {} 


{class num:counts} = {} 


{class num:counts} = {2.0: 1} 

class num = 2 , class_name = penguin

{class num:counts} = {} 


{class num:counts} = {3.0: 1} 

class num = 3 , class_name = puffin

{class num:counts} = {2.0: 3} 

class num = 2 , class_name = penguin
class num = 2 , class_name = penguin
class num = 2 , class_name = penguin

{class num:counts} = {2.0: 1} 

class num = 2 , class_name = penguin

{class num:counts} = {2.0: 3} 

class num = 2 , class_name = penguin
class num = 2 , class_name = penguin
class num = 2 , class_name = penguin

{class num:counts} = {0.0: 1} 

class num = 0 , class_name = fish

{class num:counts} = {0.0: 3} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 3, 4.0: 2} 

class num = 4 , class_name = shark
class num = 4 , class_name = shark
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 4, 4.0: 4} 

class num = 4 , class_name = shark
class num = 4 , class_name = shark
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 4 , class_name = shark
class num = 4 , class_name = shark
class num = 0 , class_name = fish

{class num:counts} = {0.0: 2, 6.0: 1} 

class num = 6 , class_name = stingray
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 2} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {} 


{class num:counts} = {1.0: 26} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {0.0: 4} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 3} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 4} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 1, 4.0: 1} 

class num = 4 , class_name = shark
class num = 0 , class_name = fish

{class num:counts} = {0.0: 2} 

class num = 0 , class_name = fish
class num = 0 , class_name = fish

{class num:counts} = {0.0: 1} 

class num = 0 , class_name = fish

{class num:counts} = {} 


{class num:counts} = {} 


{class num:counts} = {1.0: 2} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {6.0: 1} 

class num = 6 , class_name = stingray

{class num:counts} = {1.0: 32} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish

{class num:counts} = {1.0: 2} 

class num = 1 , class_name = jellyfish
class num = 1 , class_name = jellyfish
```