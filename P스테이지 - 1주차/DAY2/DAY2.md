# 프로젝트 회고

1. Competition을 위한 베이스라인 작성
    - 어제 작성한 베이스라인에서 라벨링 코드에 오류가 존재 -> 수정
    - VGG11 모델 학습 결과 정확도가 낮았던 이유도 라벨링 코드에 문제가 있었기 때문.
    - efficient net - b4, epoch 5 로 학습한 결과 : `정확도 : 0.49`
    
2. Custom Dataset
    - 항상 pytorch를 공부할 때 CustomDataset 이 잘 이해되지 않아서 힘들었는데, 이제는 좀 이해가됨
    - 처음 Custom Dataset의 경우 image를 cv2로 불러왔었는데, augementation 적용시 에러 발생 
    - Custom Dataset에서 image 를 PIL로 읽는 것으로 수정함.
    - 라벨링 코드를 검증해보지 않았는데 오류투성이 .. 즉각 수정..

3. 그외
    - 데이터 셋에서 60대 이상의 비율이 너무 적음. 어떻게 오버샘플링을 해야 할 지 고민중.
    - 클래스에 weight를 주거나, cutMix를 사용해 볼 예정
    - 오늘 피어세션에서 조원이 CutMix사용 해봤는데 성능 향상 그닥 ? 그래서 이것도 사실 고민중.

    
<br/>
<br/>

# CustomDataset 코드 
---

```python
# Image 불러오는거
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import numpy as np
from glob import glob

class MaskDataset(Dataset):
  def __init__(self, data_root, is_Train=True, input_size=255, transform=None):
    super(MaskDataset, self).__init__()

    self.img_list = self._load_img_list(data_root, is_Train)
    self.len = len(self.img_list)
    self.input_size = input_size
    self.transform = transform

  def __getitem__(self, index):
    # 이미지로드
    img =Image.open(self.img_list[index])


    if self.transform:
      img = self.transform(img)

    # Ground Truth
    label = self._get_class_idx_from_img_name(self.img_list[index])

    return img, label

  def __len__(self):
    return self.len

  def _load_img_list(self, data_root, is_Train):
    
    # 폴더이름 1234-1 -> 12341 로수정
    full_img_list = glob(data_root + '/*')
    for dir in full_img_list:
      dirname = os.path.basename(dir)
      if '-1' in dirname:
        os.rename(dir, dir.replace(dirname, dirname.replace('-1', '1')))
      elif '-2' in dirname:
        os.rename(dir, dir.replace(dirname, dirname.replace('-2','2')))
    
    # ID < 6000 -> train
    # 6200 < ID -> test
    img_list = []
    for dir in glob(data_root + '/*'):
      if is_Train and (self._load_img_ID(dir) < 6200):
        img_list.extend(glob(dir+'/*'))
      elif not is_Train and (6199 < self._load_img_ID(dir)):
        img_list.extend(glob(dir+'/*'))

    return img_list

  def _load_img_ID(self, img_path):
    return int(os.path.basename(img_path).split('_')[0])

  def _get_class_idx_from_img_name(self, img_path):
    img_name = os.path.basename(img_path)
    img_age = img_path.split('/')[-2].split('_')[-1]
    img_gender = img_path.split('/')[-2].split('_')[1]
    cnt = 0
    
    if 'incorrect' in img_name: 
        cnt+= 12
        
    elif 'normal' in img_name: 
        
        cnt += 6
    elif 'mask' in img_name: 
        cnt+=cnt
        
        
    
    if int(img_age) < 30 :
        cnt = cnt
    elif int(img_age) < 60 :
        cnt +=1
    else :
        cnt +=2
        
    
    if 'female' in img_gender:
        cnt+=3
        
    
        
    return cnt
    
```