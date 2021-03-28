# Manga Face Clustering
Official PyTorch **reimplementation** of "Adaptation of Manga Face Representation for Accurate Clustering" presented in SIGGRAPH Asia 2018.

## Environment

```bash
pip install -r requirements.txt
```

## How to Use
### Dataset Preparation
Download the Manga109 dataset from [the official website](http://www.manga109.org/en/index.html).
The annotation version that we used is `v2018.05.31`.
You can see the version list [here](http://www.manga109.org/en/annotations.html#stats).

```bash
# script/split_dataset.py /path/to/Manga109_20xx_xx_xx
python script/crop_faces.py /path/to/Manga109_20xx_xx_xx
# script/split_dataset.py /path/to/Manga109_20xx_xx_xx
```
The commands that are commented out have already been executed.

The statistics of the test data can be obtained using the below command.

```bash
python script/get_data_stats.py /path/to/Manga109_20xx_xx_xx
```

### Pre-training
If you want to pre-train a model by yourself, execute the following command.

```bash
python pre_train.py /path/to/Manga109_20xx_xx_xx
# evaluation of a pre-trained model
python eval.py /path/to/Manga109_20xx_xx_xx --model_path results/model.pth
```

If not, download our pre-trained model from [here](https://github.com/fujibo/manga-face-clustering/releases/download/pre/model.pth) and put it to `results/model.pth`.

### Fine-tuning
Set a value in `[0, 10]` as an argument of `--title_idx`.

```bash
python train.py /path/to/Manga109_20xx_xx_xx --title_idx 0
```

## Performance

| Method | Accuracy | NMI |
| :--- | :---: | :---: |
| Pre-train (paper) | 0.48 | 0.63 |
| Pre-train (reimpl.) | 0.472 | 0.615 |
| Fine-tune (paper) | 0.64 | 0.71 |
| Fine-tune (reimpl.) | 0.666 | 0.718 |

## Difference from the Original Implementation
- We implemented with PyTorch instead of Chainer. This is because ResNet-50 pre-trained on ImageNet for Chainer is not publicly available.
- The size of the dataset for pre-training (67,336) is somewhat larger than the value in our paper (67,328). 
- The size of BEMADER_P (1,111 + 82) is somewhat larger than the value in our paper (1,105 + 82).

## Links
- Our paper: https://dl.acm.org/citation.cfm?id=3283319
- manga109api: https://github.com/manga109/manga109api
