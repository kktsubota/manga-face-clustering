import os

import manga109api
from PIL import Image
import torch.utils.data


class MangaDataset(torch.utils.data.Dataset):
    """manga dataset
    Args:
        manga109_root: manga109 directory
        titles: list of title (str)
        data_root: cropped face directory (str)
        exclude_others: remove others or not
        threshold: the number of appearance of each chara_ids
    """

    def __init__(
        self,
        manga109_root: str,
        titles,
        data_root: str = "dataset",
        exclude_others: bool = True,
        threshold: int = 0,
        transform=None,
    ) -> None:
        self.root = manga109_root
        self.data_root = data_root
        self.titles = titles
        self.transform = transform

        chara_others = set()
        with open("dataset/others_ids.txt") as f:
            for line in f:
                chara_others.add(line.rstrip())
        # obtain list of character ids and file paths
        chara_ids = list()
        self.paths = list()
        self.manga109_parser = manga109api.Parser(self.root)
        for title in self.titles:
            # We used the old version.
            annot = self.manga109_parser.get_annotation(
                title, annotation_type="annotations.v2018.05.31"
            )
            for chara in annot["character"]:
                if exclude_others and chara["@id"] in chara_others:
                    continue

                chara_path = os.path.join(self.data_root, "images", chara["@id"])
                if not os.path.exists(chara_path):
                    continue
                paths = os.listdir(chara_path)
                if len(paths) < threshold:
                    continue
                self.paths += paths
                chara_ids += [chara["@id"]] * len(paths)

        self.classes = sorted(set(chara_ids))
        self.labels = [self.classes.index(id) for id in chara_ids]
        assert len(self.labels) == len(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        label = self.labels[i]
        path = self.paths[i]
        # RGB -> BGR
        img = Image.open(
            os.path.join(self.data_root, "images", self.classes[label], path)
        ).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
