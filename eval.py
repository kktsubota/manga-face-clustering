import argparse
import os

import pandas as pd
import torch
import torch.utils.data
from torch import nn
from torchvision import models
from torchvision import transforms

from data import MangaDataset
from utils import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manga109_root", help="/path/to/Manga109_20xx_xx_xx")
    parser.add_argument("--data_root", default="dataset/")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--out", default="results/pre-train.csv")
    args = parser.parse_args()

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 1031)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    model.cuda()

    titles = list()
    with open(os.path.join(args.data_root, "test_titles.txt")) as f:
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

    score_dict = dict(accuracy=list(), nmi=list())
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
        score = evaluate(model, val_dloader, args.fast)
        print(title, "Accuracy: {:.3f}, NMI: {:.3f}".format(score["accuracy"], score["nmi"]))
        score_dict["accuracy"].append(score["accuracy"])
        score_dict["nmi"].append(score["nmi"])

    df = pd.DataFrame.from_dict(score_dict)
    df.index = titles
    df.to_csv(args.out)
    print(df)
    print(df.mean())


if __name__ == "__main__":
    main()
