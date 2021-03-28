import argparse
import logging
import os
from pathlib import Path

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn
import torch.utils.data
from torchvision import models
from torchvision import transforms

from data import MangaDataset
from transforms import RandomGammaCorrection
from utils import evaluate


torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser(description="pre-train")
    parser.add_argument("manga109_root", help="/path/to/Manga109_20xx_xx_xx")
    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--batchsize", "-b", type=int, default=64)
    parser.add_argument("--epoch", "-e", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="results", type=Path)
    args = parser.parse_args()

    args.out.mkdir(exist_ok=True, parents=True)

    # logging
    logger = logging.getLogger(__name__)
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(filename=(args.out / "log"), level=logging.DEBUG, format=fmt)
    logger.info(args)

    train_titles = list()
    val_titles = list()
    with open(os.path.join(args.data_root, "train_titles.txt")) as f:
        for line in f:
            train_titles.append(line.rstrip())
    with open(os.path.join(args.data_root, "val_titles.txt")) as f:
        for line in f:
            val_titles.append(line.rstrip())

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomGammaCorrection(),
            transforms.Normalize(mean, std),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_data = MangaDataset(
        args.manga109_root,
        train_titles,
        args.data_root,
        exclude_others=True,
        threshold=10,
        transform=train_transform,
    )
    val_data = MangaDataset(
        args.manga109_root,
        val_titles[-1:],
        args.data_root,
        exclude_others=True,
        transform=val_transform,
    )
    num_classes = len(train_data.classes)
    logger.info("train_size: {}".format(len(train_data)))
    logger.info("train_class: {}".format(num_classes))
    logger.info("val_size: {}".format(len(val_data)))
    logger.info("val_class: {}".format(len(val_data.classes)))

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, num_classes)
    criterion = nn.CrossEntropyLoss()
    model.cuda()
    criterion.cuda()

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    train_dloader = torch.utils.data.DataLoader(
        train_data, args.batchsize, shuffle=True, num_workers=4
    )
    val_dloader = torch.utils.data.DataLoader(
        val_data, args.batchsize, shuffle=False, num_workers=4, drop_last=False
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.epoch // 2, gamma=0.1
    )

    nmi: float = evaluate(model, val_dloader, fast=True)["nmi"]
    logger.info("NMI = {}".format(nmi))

    for epoch in range(args.epoch):
        model.train()
        for i, (img, label) in enumerate(train_dloader):
            loss = criterion(model(img.cuda()), label.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                logger.info(
                    "[{}] {} / {}, lr={:.6f}, loss={}".format(
                        epoch,
                        args.batchsize * i,
                        len(train_data),
                        optimizer.param_groups[0]["lr"],
                        loss.item(),
                    )
                )
        else:
            scheduler.step()
            nmi: float = evaluate(model, val_dloader, fast=True)["nmi"]
            logger.info("NMI = {}".format(nmi))

        if epoch % 20 == 0:
            weights = model.state_dict()
            for key in weights.keys():
                weights[key] = weights[key].cpu()
            torch.save(
                weights,
                (args.out / "model_ep{:03d}.pth".format(epoch)),
            )

    torch.save(model.cpu().state_dict(), (args.out / "model.pth"))


if __name__ == "__main__":
    main()
