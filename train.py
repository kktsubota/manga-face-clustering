import argparse
from collections import defaultdict
import itertools
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

from data import MangaFTDataset, MangaDataset
from utils import extract_features, evaluate


torch.backends.cudnn.benchmark = True


class ContrastiveLoss(nn.Module):
    """Constrastive Loss following chainer implementation"""

    def __init__(self, tau: float):
        super().__init__()
        self.tau = tau

    def forward(self, x0, x1, y):
        dist_pos = (x0 - x1).square().sum(dim=1)
        dist_neg = torch.relu(self.tau - dist_pos.sqrt()).square()
        loss = (y * dist_pos + (1 - y) * dist_neg) * 0.5
        return loss.mean()


class PairwiseLabel(object):
    VMIN = -10

    def __init__(self, length: int) -> None:
        super().__init__()
        self.length = length
        self.matrix = (
            torch.ones((self.length, self.length), dtype=torch.long) * self.VMIN
        )

    @torch.no_grad()
    def calc_frame_constraints(self, frames) -> torch.Tensor:
        """update pairwise labels using frames
        Args:
            frames: list of frame_id
        """
        # frame_id: list of indices of images
        frame_dict = defaultdict(list)
        for i, frame in enumerate(frames):
            if frame is None:
                continue
            frame_dict[frame].append(i)

        for indices in frame_dict.values():
            for idx0, idx1 in itertools.permutations(indices, 2):
                self.matrix[idx0, idx1] = 0
        return self.matrix == 0

    @torch.no_grad()
    def calc_page_constraints(
        self, features, pages, page_range: int = -1, top_n: int = 1
    ) -> None:
        """update pairwise labels using features and pages
        Args:
            features (torch.Tensor):
            pages: list of page (int)
            page_range (int)
            top_n (int)
        """
        similarity = features @ features.T
        similarity[torch.arange(self.length), torch.arange(self.length)] = self.VMIN

        pages = torch.Tensor(pages)
        modified_similarity = torch.ones_like(similarity) * self.VMIN
        for i in range(self.length):
            if page_range < 0:
                modified_similarity[i][:] = similarity[i][:]
            else:
                page_min = pages[i] - page_range
                page_max = pages[i] + page_range
                indices = (pages >= page_min) & (pages <= page_max)
                modified_similarity[i][indices] = similarity[i][indices]

        constraints = modified_similarity.topk(top_n, dim=1).indices.squeeze()
        for i, j in enumerate(constraints):
            self.matrix[i, j] = 1
            self.matrix[j, i] = 1
        return self.matrix == 1

    @torch.no_grad()
    def propagate_constraints(self, frame_const, page_const):
        """propagate constraints
        Args:
            frame_const: constraints by frames
            page_const: constraints by page

        >>> C_ij = 0 and C_jk = 1 -> C_ik = 0
        >>> C_ij = 1 and C_jk = 1 -> C_ik = 1
        >>> C_ii = 0
        """
        arange = torch.arange(self.length)
        page_const[arange, arange] = 1

        # negative/positive constraints by propagation
        neg_prop = torch.zeros_like(self.matrix)
        pos_prop = torch.zeros_like(self.matrix)

        for i in range(self.length):
            for j in range(self.length):
                # class of i != class of j
                # class of j == class of k
                # => class of i != class of k (including k == j)
                if frame_const[i, j]:
                    for k in range(self.length):
                        if page_const[j, k]:
                            neg_prop[i, k] = 1
                            neg_prop[k, i] = 1

                # class of i == class of j
                # class of j == class of k
                # => class of i == class of k (including i == j or j == k)
                elif page_const[i, j]:
                    for k in range(self.length):
                        if page_const[j, k]:
                            pos_prop[i, k] = 1
                            pos_prop[k, i] = 1

        self.matrix[pos_prop == 1] = 1
        self.matrix[neg_prop == 1] = 0
        self.matrix[arange, arange] = self.VMIN

    @torch.no_grad()
    def __getitem__(self, index):
        matrix = self.matrix[index][:, index]
        indices_0, indices_1 = torch.where(matrix != self.VMIN)
        # remove duplication
        use_indices = indices_0 < indices_1
        indices_0, indices_1 = indices_0[use_indices], indices_1[use_indices]
        return indices_0, indices_1, matrix[indices_0, indices_1]


def main():
    parser = argparse.ArgumentParser(description="pre-train")
    parser.add_argument("manga109_root", help="/path/to/Manga109_20xx_xx_xx")
    parser.add_argument("--data_root", default="dataset")
    parser.add_argument("--batchsize", "-b", type=int, default=64)
    parser.add_argument("--epoch", "-e", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=1.4)
    parser.add_argument("--out", default="results-ft", type=Path)
    parser.add_argument("--model_path", default="results/model.pth")
    parser.add_argument("--title_idx", type=int)
    args = parser.parse_args()

    args.out.mkdir(exist_ok=True, parents=True)

    # logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(filename=(args.out / "log"), level=logging.DEBUG, format=fmt)
    logger.info(args)

    test_titles = list()
    with open(os.path.join(args.data_root, "test_titles.txt")) as f:
        for line in f:
            test_titles.append(line.rstrip())
    title = test_titles[args.title_idx]

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
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
    train_data = MangaFTDataset(
        args.manga109_root,
        title,
        args.data_root,
        exclude_others=False,
        transform=train_transform,
    )
    val_data = MangaDataset(
        args.manga109_root,
        [title],
        args.data_root,
        exclude_others=True,
        transform=val_transform,
    )
    logger.info("train_size: {}".format(len(train_data)))
    logger.info("train_class: {}".format(len(train_data.classes)))
    logger.info("val_size: {}".format(len(val_data)))
    logger.info("val_class: {}".format(len(val_data.classes)))

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 1031)
    model.load_state_dict(torch.load(args.model_path))
    criterion = ContrastiveLoss(args.tau)
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

    nmi = evaluate(model, val_dloader, fast=True)["nmi"]
    logger.info("NMI = {}".format(nmi))

    # extract features of the target dataset
    init_dloader = torch.utils.data.DataLoader(
        train_data, args.batchsize, shuffle=False, num_workers=4
    )
    transform = train_data.transform
    train_data.transform = val_transform
    model.eval()
    with torch.no_grad():
        features = list()
        for img, label, _ in init_dloader:
            feature = extract_features(model, img.cuda())
            features.append(feature)
        features = torch.cat(features, dim=0)
    train_data.transform = transform

    frames = [train_data.get_frame_id(i) for i in range(len(train_data))]
    pages = [train_data.get_page(i) for i in range(len(train_data))]

    pairwise_label = PairwiseLabel(len(train_data))
    frame_constraints = pairwise_label.calc_frame_constraints(frames)
    page_constraints = pairwise_label.calc_page_constraints(
        features, pages, page_range=1, top_n=1
    )
    pairwise_label.propagate_constraints(frame_constraints, page_constraints)

    for epoch in range(args.epoch):
        model.train()
        for img, _, index in train_dloader:
            indices_0, indices_1, label = pairwise_label[index]
            if len(label) == 0:
                continue
            feature = extract_features(model, img.cuda())
            loss = criterion(feature[indices_0], feature[indices_1], label.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            logger.info(
                "[{}] {} lr={:.6f}, loss={}".format(
                    epoch,
                    len(train_data),
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                )
            )
            scheduler.step()
            nmi: float = evaluate(model, val_dloader, fast=True)["nmi"]
            logger.info("NMI = {}".format(nmi))

    logger.info(evaluate(model, val_dloader, fast=False))
    torch.save(
        model.cpu().state_dict(), (args.out / "model-{}.pth".format(args.title_idx))
    )


if __name__ == "__main__":
    main()
