# 프로젝트 회고

1. Competition을 위한 코드 진행 과정

    - 기존 Custom Dataset을 갈아엎음
    - 기존에는 Custom dataset에는 아래와 같이 `img_path` 를 파라미터로 넘겨주면, class 내부에서 라벨을 만들고 리턴 하는 형식이였지만
    - 그렇게하니 train data 와 validation data 를 나누는게 애매해져서 그냥 class 외부로 뺴주고 'img_path' 와 'label'을 파라미터로 넘겨주는 형식으로 수정.

```python
# Img로 불러오는거
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os
import cv2
import numpy as np
from glob import glob


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환합니다."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.FloatTensor(image),
                'label': torch.FloatTensor(label)}
# to_tensor 선언

to_tensor = transforms.Compose([
                      ToTensor()
])
class MaskDataset(Dataset):
  def __init__(self, img_list, label, transforms=to_tensor, augmentations=None):
    super(MaskDataset, self).__init__()

    self.img_list = img_list
    self.len = len(self.img_list)
    self.label = label
    self.transforms = transforms
    self.augmentations=augmentations
  def __getitem__(self, index):
    img_path = self.img_list[index]
    #img =Image.open(self.img_list[index])
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img/255).astype('float32')
    # Ground Truth
    label = self.label[index]
    label = np.array(label).astype(float)
    sample={'image':img,'label':label}
    
    if self.augmentations:
      sample['image'] = self.augmentations(image=sample['image'])['image']
    if self.transforms:
      sample = self.transforms(sample)
    

    img = sample['image']
    label = sample['label']

    return img, label

  def __len__(self):
    return self.len

```python
def get_img_list(data_root):
    
    full_img_list = glob(data_root + '/*')
    
    for dir in full_img_list:
        dirname = os.path.basename(dir)
        if '-1' in dirname:
            os.rename(dir, dir.replace(dirname, dirname.replace('-1', '1')))
        elif '-2' in dirname:
            os.rename(dir, dir.replace(dirname, dirname.replace('-2','2')))
    

    img_list = []
    
    for dir in glob(data_root + '/*'):
        img_list.extend(glob(dir+'/*'))
        
    print(len(img_list))
    return img_list


def get_label(img_path):
    img_name = os.path.basename(img_path)
    img_age = img_path.split('/')[-2].split('_')[-1]
    img_gender = img_path.split('/')[-2].split('_')[1]
    cnt = 0
    
    if 'incorrect' in img_name:         
        cnt+= 6
    elif 'normal' in img_name:         
        cnt += 12
    else:
        cnt = cnt       
    
    if int(img_age) < 30 :
        cnt = cnt
    elif int(img_age) < 58 :        
        cnt += 1
    else :
        cnt +=2

    if 'female' in img_gender:
        cnt+=3        
    return cnt
```
```python
val_idx = 16000

# Dataset and Data Loader

transform = A.Compose([
    A.CenterCrop(350,350)
])


train_dataset = MaskDataset(img_list[:val_idx],label[:val_idx])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = MaskDataset(img_list[val_idx:], label[val_idx:])
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

```

    - 드디어 정상적으로 작동하는 코드를 완성함.
    - 정확도가 70%를 겨우 넘겼는데 f1 score가 너무낮아서 다시 순위가 낮아짐 ㅎ ㅎ.. f1 score를 올릴 수 있는 방법을 연구하는중.
    - 오늘 오후에 모든 base line 코드가 배포되기때문에, 내 코드와 base line 코드를 조금 섞어서 완전한 코드를 완성할 계획 -> 주말
    
2. Daily mission
    - K-fold 적용 -> clear
    - TTA 적용 시켜보기 -> 보류
    
3. 그외
    - 클래스 불균형 문제 -> 58세 이미지 까지 60세로 라벨링하여 학습 (이현규 피어님 의견)
    - f1 score를 위한 loss를 찾았는데 오히려 성능이 별로 ? 왜그런지 모르겠다.