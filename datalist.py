from random import shuffle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

'''
1. 对图片进行按比例缩放
2. 对图片进行随机位置的截取
3. 对图片进行随机的水平和竖直翻转
4. 对图片进行随机角度的旋转
5. 对图片进行亮度、对比度和颜色的随机变化
'''


# 自己写Dataset至少需要有这样的格式
class Dataset(Dataset):
    def __init__(self, base_data_path='./', use_aug=True, lines=''):
        super(Dataset, self).__init__()
        self.base_path = base_data_path
        self.annotation_lines = lines
        self.train_batches = len(self.annotation_lines)
        self.use_aug = use_aug

    def __len__(self):
        return self.train_batches

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.annotation_lines)
        n = len(self.annotation_lines)
        index = index % n
        img, y = self.collect_image_label(self.annotation_lines[index])

        if self.use_aug:
            img = self.img_augment(img)
        img = img.resize((32, 32), Image.BICUBIC)
        img = np.array(img, dtype=np.float32)
        temp_img = np.transpose(img / 255.0)
        temp_y = int(y) - 1
        return temp_img, temp_y

    def collect_image_label(self, line):
        line = line.split('*')
        image_path = line[0]
        label = line[1]
        image = Image.open(image_path).convert("RGB")

        return image, label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def img_augment(self, image):
        # 随机位置裁剪
        random_crop = self.rand() < 0.5
        # 中心裁剪
        center_crop = self.rand() < 0.5
        # 填充后随机裁剪
        random_crop_padding = self.rand() < 0.5
        # 水平翻转
        h_flip = self.rand() < 0.5
        # 竖直翻转
        v_flip = self.rand() < 0.5
        # 亮度
        bright = self.rand() < 0.5
        # 对比度
        contrast = self.rand() < 0.5
        # 饱和度
        saturation = self.rand() < 0.5
        # 颜色随机变换
        color = self.rand() < 0.5
        compose = self.rand() < 0.5
        # 旋转30
        rotate = self.rand() < 0.5

        if h_flip:
            image = transforms.RandomHorizontalFlip()(image)
        if v_flip:
            image = transforms.RandomVerticalFlip()(image)
        if rotate:
            image = transforms.RandomRotation(30)(image)
        if bright:
            image = transforms.ColorJitter(brightness=1)(image)
        if contrast:
            image = transforms.ColorJitter(contrast=1)(image)
        if saturation:
            image = transforms.ColorJitter(saturation=1)(image)
        if color:
            image = transforms.ColorJitter(hue=0.5)(image)
        if compose:
            image = transforms.ColorJitter(0.5, 0.5, 0.5)(image)
        if random_crop:
            image = transforms.RandomCrop(100)(image)
        if center_crop:
            image = transforms.CenterCrop(100)(image)
        if random_crop_padding:
            image = transforms.RandomCrop(100, padding=8)(image)

        return image


if __name__ == "__main__":
    Dataset()

