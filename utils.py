import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn import functional as F

from metrics import cluster_accuracy, nmi_score


def extract_features(self: nn.Module, x: torch.Tensor):
    """extract features by resnet50"""
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = F.normalize(x, p=2, dim=1)
    return x


@torch.no_grad()
def evaluate(model, val_dloader, fast: bool = False):
    model.eval()
    features = list()
    labels = list()
    for img, label in val_dloader:
        feat = extract_features(model, img.cuda())
        features.append(feat.cpu().numpy())
        labels.append(label.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    n_init: int = 1 if fast else 1000
    kmeans = KMeans(np.unique(labels).size, n_init=n_init)
    preds = kmeans.fit_predict(features)

    return {
        "accuracy": cluster_accuracy(labels, preds),
        "nmi": nmi_score(labels, preds),
    }
