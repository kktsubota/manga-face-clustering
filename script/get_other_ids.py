import argparse

import manga109api


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("root", help="/path/to/Manga109_20xx_xx_xx")
args = arg_parser.parse_args()

others_names = ["Others", "Other", "the other", "Ｏｔｈｅｒ", "other", "Ｏｈｔｅｒ"]
# for Buraritessen
others_names += ["Other（スリ）", "Other（赤穂浪士）", "Other（役人）", "Other(大吉屋の手下）"]

others_ids = list()
parser = manga109api.Parser(args.root)
for book in parser.books:
    annot = parser.get_annotation(book, annotation_type="annotations.v2018.05.31")
    for chara in annot["character"]:
        if chara["@name"] in others_names:
            others_ids.append(chara["@id"])

with open("dataset/others_ids.txt", "w") as f:
    for id in others_ids:
        f.write(id + "\n")
