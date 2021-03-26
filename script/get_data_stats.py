import argparse
import sys

sys.path = ["./"] + sys.path
from data import MangaDataset


parser = argparse.ArgumentParser()
parser.add_argument("manga109_root")
parser.add_argument("--data_root", default="dataset/")
args = parser.parse_args()

titles = list()
with open("dataset/test_titles.txt") as f:
    for line in f:
        titles.append(line.rstrip())

for title in titles:
    val_data = MangaDataset(
        args.manga109_root,
        [title],
        args.data_root,
        exclude_others=True,
    )
    print(title, len(val_data), len(val_data.classes))
    val_data = MangaDataset(
        args.manga109_root,
        [title],
        args.data_root,
        exclude_others=False,
    )
    print(title, len(val_data), len(val_data.classes))
