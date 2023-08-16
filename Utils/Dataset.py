import torch.utils.data as data
import PIL.Image as Image

import glob
import os


class CustomDataset(data.Dataset):
    def __init__(self, dataset_dir, run_type, cls, transform):
        super(CustomDataset, self).__init__()

        self.transform = transform
        self.run_type = run_type
        self.folder = glob.glob(f'{os.path.join(dataset_dir, run_type)}/*')

    def __getitem__(self, index):
        if self.run_type=='train':
            v1 = self.transform(Image.open(self.folder[index]))
            v2 = self.transform(Image.open(self.folder[index]))

            return [v1, v2]
        else:
            v1 = self.transform(Image.open(self.folder[index]))

            return v1

    def __len__(self):
        return len(self.folder)