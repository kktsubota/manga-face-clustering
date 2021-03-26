import argparse
from pathlib import Path

import tqdm
import cv2
import manga109api


def expand_bbox(bbox, page_width: int, page_height: int):
    xmin, ymin, xmax, ymax = bbox
    x_margin = (xmax - xmin) // 2
    y_margin = (ymax - ymin) // 2

    xmin = max(xmin - x_margin, 0)
    xmax = min(xmax + x_margin, page_width)
    ymin = max(ymin - y_margin, 0)
    ymax = min(ymax + y_margin, page_height)
    return xmin, ymin, xmax, ymax


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("manga109dir")
    arg_parser.add_argument("--out", default="dataset/images", type=Path)
    args = arg_parser.parse_args()

    parser = manga109api.Parser(root_dir=args.manga109dir)
    books = list()
    for split in {"train", "val", "test"}:
        with open("dataset/{}_titles.txt".format(split)) as f:
            for line in f:
                books.append(line.rstrip())

    for book in tqdm.tqdm(books):
        annotation = parser.get_annotation(book=book, annotation_type="annotations.v2018.05.31")
        for page in annotation["page"]:
            name = "{:03d}.jpg".format(page["@index"])
            path = parser.root_dir / "images" / annotation["title"] / name
            # H, W, C
            page_img = cv2.imread(path.as_posix())
            for face in page["face"]:
                (args.out / face["@character"]).mkdir(exist_ok=True, parents=True)

                # small face images are excluded.
                if (face["@xmax"] - face["@xmin"]) <= 30:
                    continue
                if (face["@ymax"] - face["@ymin"]) <= 30:
                    continue
                attrs = ("@xmin", "@ymin", "@xmax", "@ymax")
                bbox = [face[attr] for attr in attrs]
                bbox = expand_bbox(bbox, page["@width"], page["@height"])
                img = page_img[bbox[1] : bbox[3], bbox[0] : bbox[2], :]

                # NOTE: ".png" should be used
                save_path = args.out / face["@character"] / (face["@id"] + ".jpg")
                cv2.imwrite(save_path.as_posix(), img)


if __name__ == "__main__":
    main()
