import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from Utils.Options import Param


class KNN(Param):
    def __init__(self, model, k):
        super(KNN, self).__init__()

        self.k = k
        self.model = model.to(self.device)
        self.model.eval()

    def feature_extract(self, loader):
        x_lst = []
        features = []
        label_lst = []

        with torch.no_grad():
            for idx, (item, label) in enumerate(loader):
                h = self.model(item.to(self.device))
                features.append(h)

                x_lst.append(item)
                label_lst.append(label)

            x_total = torch.stack(x_lst)
            h_total = torch.stack(features)
            label_total = torch.stack(label_lst)

        return x_total, h_total, label_total

    def knn(self, features, labels, k=1):
        feature_dim = features.shape[-1]
        with torch.no_grad():
            features_np = features.cpu().view(-1, feature_dim).numpy()
            labels_np = labels.cpu().view(-1).numpy()

            self.cls = KNeighborsClassifier(k, metric='cosine').fit(features_np, labels_np)
            acc = self.eval(features, labels)

        return acc

    def eval(self, features,  labels):
        feature_dim = features.shape[-1]
        features = features.cpu().view(-1, feature_dim).numpy()
        labels = labels.cpu().view(-1).numpy()

        acc = 100 * np.mean(cross_val_score(self.cls, features, labels))

        return acc

    def fit(self, train_loader, test_loader=None):
        with torch.no_grad():
            x_train, h_train, l_train = self.feature_extract(train_loader)
            train_acc = self.knn(h_train, l_train, self.k)

            if test_loader is not None:
                x_test, h_test, l_test = self.feature_extract(test_loader)
                test_acc = self.eval(h_test, l_test)

                return train_acc, test_acc

        return train_acc