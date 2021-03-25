import argparse

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.utils.data
from torch import nn
from torchvision import models
from torchvision import transforms

from data import MangaDataset
from pre_train import extract_features
from metrics import cluster_accuracy, nmi_score


def evaluate(model, val_dloader):
    model.eval()
    features = list()
    labels = list()
    for img, label in val_dloader:
        with torch.no_grad():
            feat = extract_features(model, img.cuda())
            features.append(feat.cpu().numpy())
            labels.append(label.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    kmeans = KMeans(np.unique(labels).size, n_init=1000)
    preds = kmeans.fit_predict(features)

    return cluster_accuracy(labels, preds), nmi_score(labels, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manga109_root", help="/path/to/Manga109_20xx_xx_xx")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--model_path", default=None)
    args = parser.parse_args()

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 1031)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    model.cuda()

    titles = list()
    with open("dataset/test_titles.txt") as f:
        for line in f:
            titles.append(line.rstrip())

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    for title in titles:
        val_data = MangaDataset(
            args.manga109_root,
            [title],
            args.data_root,
            exclude_others=True,
            transform=val_transform,
        )
        val_dloader = torch.utils.data.DataLoader(
            val_data, 50, shuffle=False, num_workers=4, drop_last=False
        )
        acc, nmi = evaluate(model, val_dloader)
        print(title, "Accuracy: {:.3f}, NMI: {:.3f}".format(acc, nmi))


if __name__ == "__main__":
    main()
