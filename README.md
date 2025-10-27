## Overview
**Lordre** is a conceptually novel, efficient, and fully convolutional framework for real-time instance segmentation.
In contrast to region boxes or anchors (centers), Lordre adopts a sparse set of **instance activation maps** as object representation, to highlight informative regions for each foreground objects.
Then it obtains the instance-level features by aggregating features according to the highlighted regions for recognition and segmentation.
The bipartite matching compels the instance activation maps to predict objects in a one-to-one style, thus avoiding non-maximum suppression (NMS) in post-processing. 

## Models
We provide two versions of Lordre, *i.e.*, the basic IAM (3x3 convolution) and the Group IAM (G-IAM for short), with different backbones.

**Note:** 
* **I will continue adding more models** including more efficient convolutional networks, vision transformers, and larger models for high performance and high speed, please stay tuned;!
* *input* denotes the shorter side of the input, *e.g.*, 512x864 and 608x864, we keep the aspect ratio of the input and the longer side is no more than 864.
* The inference speed might slightly change on different machines (2080 Ti) and different versions of detectron (we mainly use [v0.3](https://github.com/facebookresearch/detectron2/tree/v0.3)). If the change is sharp, e.g., > 5ms, please feel free to contact us.
* For `aug` (augmentation), we only adopt the simple random crop (crop size: [384, 600]) provided by detectron2.
* We adopt `weight decay=5e-2` as default setting, which is slightly different from the original paper.

## Installation and Prerequisites

This project is built upon the excellent framework and you should install detectron2 first, please check for more details.

Install the detectron2:

```bash
git clone https://github.com/facebookresearch/detectron2.git
# if you swith to a specific version, e.g., v0.3 (recommended) or v0.6
git checkout tags/v0.6
# build detectron2
python setup.py build develop
```

## Getting Start


### &#128293; SparseInst with FP16

Lordre with FP16 achieves 30% faster inference speed and saves much training memory, we provide some comparisons about the memory, inference speed, and training speed in the below table.

|  FP16 | train mem.(log) | train mem.(`nvidia-smi`) | train speed | infer. speed | 
| :---: | :-------------: | :----------------------: | :---------: | :----------: |
| &#x2718; | 6.0G | 10.5G | 0.8690s/iter | 52.17 FPS |
| &#10003; | 3.9G | 6.8G  | 0.6949s/iter | 67.57 FPS |

Note: statistics are measured on NVIDIA 3090. With FP16, we have faster training speed and can also increase the batch size for better performance.


* Testing with FP16: enable FP16 for inference by adding `--fp16`.

```bash
python tools/test_net.py --config-file configs/sparse_inst_r50_giam_fp16.yaml --fp16 MODEL.WEIGHTS model_final.pth 
```

### Testing SparseInst

Before testing, you should specify the config file `<CONFIG>` and the model weights `<MODEL-PATH>`. In addition, you can change the input size by setting the `INPUT.MIN_SIZE_TEST` in both config file or commandline.

* [Performance Evaluation] To obtain the evaluation results, *e.g.*, mask AP on COCO, you can run:

```bash
python tools/train_net.py --config-file <CONFIG> --num-gpus <GPUS> --eval MODEL.WEIGHTS <MODEL-PATH>
# example:
python tools/train_net.py --config-file configs/sparse_inst_r50_giam.yaml --num-gpus 8 --eval MODEL.WEIGHTS sparse_inst_r50_giam_aug_2b7d68.pth
```

* [Inference Speed] To obtain the inference speed (FPS) on one GPU device, you can run:

```bash
python tools/test_net.py --config-file <CONFIG> MODEL.WEIGHTS <MODEL-PATH> INPUT.MIN_SIZE_TEST 512
# example:
python tools/test_net.py --config-file configs/sparse_inst_r50_giam.yaml MODEL.WEIGHTS sparse_inst_r50_giam_aug_2b7d68.pth INPUT.MIN_SIZE_TEST 512
```

**Note:** 
* The [`tools/test_net.py`](./tools/test_net.py) only supports **1 GPU** and **1 image per batch** for measuring inference speed.
* The inference time consists of the *pure forward time* and the *post-processing time*. While the evaluation processing, data loading, and pre-processing for wrappers (*e.g.*, ImageList) are not included.
* `COCOMaskEvaluator` is modified from [`COCOEvaluator`](https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py) for evaluating mask-only results.

### FLOPs and Parameters

The [`get_flops.py`](tools/get_flops.py) is built based on `detectron2` and `fvcore`. 

```bash
python tools/get_flops.py --config-file <CONFIG> --tasks parameter flop
```


### Training Lordre

To train the SparseInst model on COCO dataset with 8 GPUs. 8 GPUs are required for the training. If you only have 4 GPUs or GPU memory is limited, it doesn't matter and you can reduce the batch size through `SOLVER.IMS_PER_BATCH` or reduce the input size. If you adjust the batch size, learning schedule should be adjusted according to the linear scaling rule.

```bash
python tools/train_net.py --config-file <CONFIG> --num-gpus 8 
# example
python tools/train_net.py --config-file configs/sparse_inst_r50vd_dcn_giam_aug.yaml --num-gpus 8
```
