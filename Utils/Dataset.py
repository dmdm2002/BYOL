import torch.utils.data as data
import PIL.Image as Image

import glob
import os


class CustomDataset(data.Dataset):
    def __init__(self, dataset_dir, run_type, cls, transform):
        super(CustomDataset, self).__init__()

        self.transform = transform
        self.run_type = run_type

        folder_A = glob.glob(f'{os.path.join(dataset_dir, run_type, cls[0])}/*')
        folder_B = glob.glob(f'{os.path.join(dataset_dir, run_type, cls[1])}/*')
        self.image_path = []

        for i in range(len(folder_A)):
            self.image_path.append([folder_A[i], 0])

        for i in range(len(folder_B)):
            self.image_path.append([folder_B[i], 1])

    def __getitem__(self, index):
        if self.run_type=='train':
            v1 = self.transform(Image.open(self.image_path[index][0]))
            v2 = self.transform(Image.open(self.image_path[index][0]))
            label = self.image_path[index][1]

            return [v1, v2]
        else:
            v1 = self.transform(Image.open(self.folder[index]))
            label = self.image_path[index][1]

            return [v1, label]

    def __len__(self):
        return len(self.image_path)