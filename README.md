
## Introduction
This repository provides an integrated process for training and evaluating multiple crowd counting models, each with their own original implementations and licenses.

#### Key features:
- Multi-gpu training support
- YAML-based model config files
- Modular structure for easy addition of new models

## Supported Models
| Model       | License       | Source |
|-------------|---------------|--------|
| APGCC       | MIT  | [APGCC](https://github.com/AaronCIH/APGCC) |
| CLIP-EBC    | MIT  | [CLIP-EBC](https://github.com/Yiming-M/CLIP-EBC) |
| CLTR        | MIT  | [CLTR](https://github.com/dk-liang/CLTR) |
| DMCount     | MIT  | [DMCount](https://github.com/cvlab-stonybrook/DM-Count) |
| FusionCount | MIT  | [FusionCount](https://github.com/Yiming-M/FusionCount) |
| STEERER     | MIT  | [STEERER](https://github.com/taohan10200/STEERER) |
| FFNet     | MIT  | [FFNet](https://github.com/erdongsanshi/Fuss-Free-structure) |

## Supported Datasets
- ShanghaiTech A & B
- NWPU
- UCF-QNRF
- JHU-Crowd

## Installation
### 1. Clone repository
```
https://github.com/standfsk/crowd-counting-framework.git
cd crowd-counting-framework
```
### 2. Install dependencies
```
pip install -r requirements.txt
```

## Prepare dataset
```
cd datasets
python prepare.py
```

## Train
```
python train.py --save-path train --network apgcc
```

## Test
```
python test.py --save-path test --network apgcc --checkpoint output/train/best.pt --device 0 --save --log 
```

## Export
```
python export.py --save-path apgcc.onnx --network apgcc --backbone vgg16_bn --checkpoint output/train/best.pt 
```

## Acknowledgement
This project builds upon the work of many researchers in the field of crowd counting.<br>
Full credit goes to original authors of supported models
- [APGCC](https://github.com/AaronCIH/APGCC)
- [CLIP-EBC](https://github.com/Yiming-M/CLIP-EBC)
- [CLTR](https://github.com/dk-liang/CLTR)
- [DMCount](https://github.com/cvlab-stonybrook/DM-Count)
- [FusionCount](https://github.com/Yiming-M/FusionCount)
- [STEERER](https://github.com/taohan10200/STEERER)
- [FFNet](https://github.com/erdongsanshi/Fuss-Free-structure)




