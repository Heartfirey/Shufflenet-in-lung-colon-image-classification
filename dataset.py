import os

import numpy as np
from PIL import Image
from random import shuffle
from torch.utils.data import Dataset
from torch import from_numpy
from torchvision import transforms

label_dict = {
    'colon_aca': 1,
    'colon_n': 2,
    'lung_aca': 3,
    'lung_n': 4,
    'lung_scc': 5, }


class SelfDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(base_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = from_numpy(np.array(Image.open(path_img).convert("RGB").resize((32, 32)))).float().permute(2, 0, 1)

        return np.array(img), label

    @staticmethod
    def get_img_info(data_dir):
        data_info = []
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpeg'), img_names))

                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label_idx = label_dict[sub_dir]
                    label = [0.0, 0.0, 0.0, 0.0, 0.0]
                    label[label_idx - 1] = 1.0
                    data_info.append((path_img, np.array(label)))
        return data_info


# if __name__ == "__main__":
#     ds = SelfDataset('.\\lung_colon_image_set\\', None)
