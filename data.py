import os

import manga109api
import numpy as np
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
        with open(os.path.join(self.data_root, "others_ids.txt")) as f:
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
        img = Image.open(
            os.path.join(self.data_root, "images", self.classes[label], path)
        ).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class MangaFTDataset(MangaDataset):
    """manga dataset for fine-tuning
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
        title,
        data_root: str = "dataset",
        exclude_others: bool = True,
        threshold: int = 0,
        transform=None,
    ) -> None:
        super().__init__(
            manga109_root, [title], data_root, exclude_others, threshold, transform
        )
        # re-run
        annot = self.manga109_parser.get_annotation(
            title, annotation_type="annotations.v2018.05.31"
        )

        bbox_attr = ["@xmin", "@ymin", "@xmax", "@ymax"]
        # face_id -> page number (int)
        self.page_dict = dict()
        # face_id -> frame_id (str)
        self.frame_dict = dict()
        for page in annot["page"]:
            frame_ids = list()
            frame_bboxes = list()
            for frame in page["frame"]:
                frame_ids.append(frame["@id"])
                bbox = [frame[attr] for attr in bbox_attr]
                frame_bboxes.append(bbox)
            frame_bboxes = np.array(frame_bboxes)

            for face in page["face"]:
                bbox = [face[attr] for attr in bbox_attr]
                bbox = np.array(bbox)[np.newaxis]
                ir = get_intersect_ratio(bbox, frame_bboxes)
                idx = np.argmax(ir, axis=1).squeeze()
                self.frame_dict[face["@id"]] = frame_ids[idx]
                self.page_dict[face["@id"]] = page["@index"]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        label = self.labels[i]
        path = self.paths[i]
        img = Image.open(
            os.path.join(self.data_root, "images", self.classes[label], path)
        ).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, i

    def get_frame_id(self, i):
        path = self.paths[i]
        face_id, _ = os.path.splitext(path)
        return self.frame_dict[face_id]

    def get_page(self, i):
        path = self.paths[i]
        face_id, _ = os.path.splitext(path)
        return self.page_dict[face_id]


def get_intersect_ratio(bbox_a, bbox_b):
    # modifiled from chainercv/utils/bbox/bbox_iou.py
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    return area_i / area_a[:, None]
