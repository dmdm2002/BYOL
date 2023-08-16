import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

import tqdm

from Model.BYOL import BYOL
from Model.KNN import KNN
from Utils.Options import Param
from Utils.Dataset import CustomDataset


class Trainer(Param):
    def __init__(self):
        super(Trainer, self).__init__()

    def run(self):
        net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = BYOL(net=net).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop((224, 224)),
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        tr_dataset = CustomDataset(self.db_path, run_type='train', cls=['cats'], transform=transform)
        tr_loader = DataLoader(dataset=tr_dataset, batch_size=self.batchsz, shuffle=True)

        te_dataset = CustomDataset(self.db_path, run_type='test_set', cls=['cats'], transform=transform)
        te_loader = DataLoader(dataset=tr_dataset, batch_size=self.batchsz, shuffle=False)

        for ep in range(0, self.full_epoch):
            total_loss = 0.
            num_batches = len(tr_loader)

            model.train()
            for idx, (v1, v2) in enumerate(tqdm.tqdm(tr_loader, desc=f'Train {ep}/{self.full_epoch}')):
                v1 = v1.to(self.device)
                v2 = v2.to(self.device)

                optimizer.zero_grad()

                loss = model(v1, v2)
                loss.backward()
                optimizer.step()

                # EMA update
                model.update_moving_average()
                total_loss += loss.item()

            total_loss_avg = total_loss/num_batches

            print(f'Train epoch: {ep}')
            print(f'total loss: {total_loss_avg}')

            if self.do_knn:
                knn_model = KNN(model, k=2)
                knn_acc = knn_model.fit(te_loader)

                print(f'KNN test: {knn_acc}')
