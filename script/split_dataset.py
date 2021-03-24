import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("manga109dir")
args = parser.parse_args()

os.makedirs("dataset", exist_ok=True)
titles = list()
for title in os.listdir(os.path.join(args.manga109dir, "images")):
    if "_vol01" in title:
        continue
    titles.append(title)
titles = sorted(titles)

titles_dict = dict()
titles_dict["test"] = titles[0:10] + [titles[43]]
titles = titles[10:43] + titles[44:]
titles_dict["val"] = titles[:10]
titles_dict["train"] = titles[10:]

for split in {"train", "val", "test"}:
    with open("dataset/{}.txt".format(split), "w") as f:
        for title in titles_dict[split]:
            f.write(title + "\n")
