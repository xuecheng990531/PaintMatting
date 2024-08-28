import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

class matting_datasets(Dataset):
    def __init__(self, data_root: str, mode: str,isnorm:str):
        super(matting_datasets, self).__init__()
        self.norm=isnorm
        print('image norm is',self.norm)
        assert os.path.exists(data_root), f"path '{data_root}' does not exist."
        self.imgs_dir = os.path.join(data_root, "new_image/")
        self.alpha = os.path.join(data_root, "alpha/")
        self.prompt = os.path.join(data_root, "mask/")
        self.trimap_path = os.path.join(data_root, "trimap/")
        self.image_names = [file for file in os.listdir(self.imgs_dir)]

        print(f'{mode}:dataset with {len(self.image_names)} examples.')
        self.mode = mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.image_names)

    def preprocess(self, image, prompt, alpha, trimap):
        if self.mode == "train":
            transform = A.Compose([
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                ToTensorV2()
            ])
            transformed = {
                'image': transform(image=image)['image'],
                'prompt': transform(image=prompt)['image'],
                'alpha': transform(image=alpha)['image'],
                'trimap': transform(image=trimap)['image'],
            }

        else:
            transform = A.Compose([
                ToTensorV2()
            ])
            transformed = {
                'image': transform(image=image)['image'],
                'prompt': transform(image=prompt)['image'],
                'alpha': transform(image=alpha)['image'],
                'trimap': transform(image=trimap)['image'],
            }

        return transformed['image'], transformed['prompt'], transformed['alpha'], transformed['trimap']

    def __getitem__(self, index):
        # 获取image和mask的路径
        image_name = self.image_names[index]
        image_path = os.path.join(self.imgs_dir, image_name)

        if image_name.endswith('.jpg'):
            alpha_path = os.path.join(self.alpha, image_name.split('.jpg')[0] + '.png')
            trimap_path = os.path.join(self.trimap_path, image_name.split('.jpg')[0] + '.png')
        else:
            alpha_path = os.path.join(self.alpha, image_name)
            trimap_path = os.path.join(self.trimap_path, image_name)

        prompt_path = os.path.join(self.prompt, image_name)

        assert os.path.exists(image_path), f"file '{image_path}' does not exist."
        assert os.path.exists(alpha_path), f"file '{alpha_path}' does not exist."
        assert os.path.exists(prompt_path), f"file '{prompt_path}' does not exist."
        assert os.path.exists(trimap_path), f"file '{trimap_path}' does not exist."

        image = cv2.imread(image_path)
        trimap = cv2.imread(trimap_path, 0)
        
        trimap[trimap>128] = 255
        trimap[trimap<5] = 0
        trimap[(trimap >= 5) & (trimap <= 128)] = 128
        
        alpha = cv2.imread(alpha_path, 0)/255.
        prompt = cv2.imread(prompt_path, 0)
        prompt[prompt>0]=255.
        prompt[prompt==0]=0
        prompt=prompt/255.
        image, prompt, alpha, trimap = self.preprocess(image, prompt, alpha, trimap)

        if self.norm:
            image=transforms.Normalize(mean=self.mean,std=self.std)(image.float())
        else:
            image=image.float()
            
        return {'image': image, 'prompt': prompt.float(), 'alpha': alpha.float(), 'trimap': trimap.float(),'image_name':image_name}


if __name__ == '__main__':
    import torch
    train_data = matting_datasets(data_root='datasets/aim500/test', mode='test',is_norm=False)
    train_dataloader = DataLoader(train_data,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=0,
                                      pin_memory=True,
                                      sampler=None,
                                      drop_last=True)
    for i,data in enumerate(train_dataloader):
        print(data['image'].dtype)
        # torch.set_printoptions(threshold=800000000)
        print(data['prompt'].dtype)
        print(data['alpha'].dtype)
        # print(data['trimap'])
        break
