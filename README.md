# One-bit Flip is All You Need: When Bit-flip Attack Meets Model Training 

## Description
This repository is the official implementation of our ICCV 2023 submission [One-bit Flip is All You Need: When Bit-flip Attack Meets Model Training](). 

## Requirement Commands (Anaconda):

Based on pytorch 1.12

Install by running the following cmd in the work directory 

```
conda create --name tfa --file ./requirements.txt
```

## Procedures

Step 1: [Download](https://www.dropbox.com/s/ax24afm1vqs9k8m/176_95.25.pth?dl=0) the model checkpoint, and then place it in the directory "checkpoint/resnet18"

Step 2: Please first fill out the path to this work directory in your server

Step 3: configure the path to CIFAR-10 dataset in config.py

Step 4: run the demo

```
python ./test_lambda.py -dc cifar10 -mc ResNet18 -bw 8 --mannual -ri 1 -ro 30
```


## Variables

### Task Specification

-bw: bit width (quantization config), 8 is provided in demo.

-mc: model choice, ResNet18 is provided.

-dc: dataset choice, cifar10 is provided.

-tn: number of target instances, no effect in demo

--rc: whether to randomly choose auxiliary samples

### Hyperparameter Specification

--mannual: whether to mannually set hyperparameters (False means using default values defined in config.py)

-bs_l: base lambda, set to 1 as default

-ri: inner ratio, lambda_in in paper

-ro: outer ratio, lambda_out in paper

-lr: learning rate for updating parameters

## Results
The log for attacking 8-bit quantized ResNet-18 is provided. Please refer to `log_resnet18_8.txt` for our results.
