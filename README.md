# 2021-VRDL-HW1

This repository contains the code for homework 1 of 2021 Fall Selected Topics in Visual Recognition using Deep Learning.

## Environment
- Python 3
- numpy 1.21.3
- timm 0.3.2
- torch 1.7.0
- torchvision 0.8.1
- tqdm 4.36.1

## Running the code
### Prepare Data
1. Download the data from [CodaLab](https://competitions.codalab.org/competitions/35668#participate)
2. `cd dataset` and `mkdir train test`
3. `unzip` training_images.zip and testing_images.zip to `train` and `test` folder, respectively

### Quick Start
1. `mkdir save`
2. Download the trained [model weight](https://drive.google.com/file/d/1oGRoMpXBeHJ_aaHL4JJLLQ27cS7SAdSK/view?usp=sharing) and put it in `save` folder
3. run `python inference.py --path ./save/cait_S36_mixup_warmup_freeze_finetune_last3`
4. the submitted answer will be saved in `answer` folder

### Train Model 
1. run `python.py` 
2. the model will be saved in `save` folder and the test result will be saved in `answer` folder

## Achnowledgements
My implementation uses the source code from the following repositories:
1. [CaiT](https://github.com/facebookresearch/deit)
2. [ResNet](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
3. [Mixup](https://github.com/facebookresearch/mixup-cifar10)
