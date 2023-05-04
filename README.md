# DistributedSGDM
Implementation of **DistributedSGDM**, as presented in:
* Distributed Momentum Methods Under Biased Gradients Estimation. Submitted to the Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023), New Orleans Ernest N. Morial Convention Center, USA, Dec 10 – Dec 16, 2023.




# Importing

> To run a new test .
```
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
```
> To aggregate CSV files.
```
import glob
import pandas as pd
import os
```
> To draw output figures.
```
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
```


# Usage
## How to Run Experiments
### Training a FCNN on [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
> The script below starts a new training process on the MNIST dataset with customized settings.
```
python MNIST_SGDM_v1.py -h

usage: MNIST_SGDM_v1.py [-h] [--seed_number SEED_NUMBER]
                        [--learning_rate LEARNING_RATE] [--beta BETA]
                        [--L2 L2] [--num_epochs NUM_EPOCHS]
                        [--num_nodes NUM_NODES] [--method METHOD]
                        [--max_norm MAX_NORM] [--top_k TOP_K]
                        [--clip_value CLIP_VALUE]

Train a neural network on MNIST

optional arguments:
  -h, --help            show this help message and exit
  --seed_number SEED_NUMBER
                        seed number
  --learning_rate LEARNING_RATE
                        learning rate for the optimizer
  --beta BETA           beta for the optimizer
  --L2 L2               weight_decay (L2 penalty)
  --num_epochs NUM_EPOCHS
                        number of epochs to train
  --num_nodes NUM_NODES
                        number of nodes
  --method METHOD       method: none, norm, clip_grad_norm, clip_grad_value,
                        Top-K
  --max_norm MAX_NORM   gradient norm clipping max value (necessary if method:
                        clip_grad_norm)
  --top_k TOP_K         number of top elements to keep in compressed gradient
                        (necessary if method: Top-K)
  --clip_value CLIP_VALUE
                        gradient clipping value (necessary if method:
                        clip_grad_value)
```
> Set --beta 1 in case you need to train DistributedSGD instead of DistributedSGDM.
> As a result of running this code; data folder, MNIST_CSV folder , and runs folder will be created.
> dataset will be downloaded in data folder.
> You can simultaneously monitor training progress through tensorboard by the files saved in runs folder.
> Training process will be loged also in a CSV file in MNIST_CSV folder.


### Training a ResNet-18 on [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist)
> The script below starts a new training process on the FashionMNIST dataset with customized settings.
```
python FashionMNIST_ResNet18_SGDM_v1.py -h

usage: FashionMNIST_ResNet18_SGDM_v1.py [-h] [--seed_number SEED_NUMBER]
                                        [--learning_rate LEARNING_RATE]
                                        [--beta BETA] [--L2 L2]
                                        [--num_epochs NUM_EPOCHS]
                                        [--num_nodes NUM_NODES]
                                        [--method METHOD]
                                        [--max_norm MAX_NORM] [--top_k TOP_K]
                                        [--clip_value CLIP_VALUE]

Train a neural network on FashionMNIST ResNet18

optional arguments:
  -h, --help            show this help message and exit
  --seed_number SEED_NUMBER
                        seed number
  --learning_rate LEARNING_RATE
                        learning rate for the optimizer
  --beta BETA           beta for the optimizer
  --L2 L2               weight_decay (L2 penalty)
  --num_epochs NUM_EPOCHS
                        number of epochs to train
  --num_nodes NUM_NODES
                        number of nodes
  --method METHOD       method: none, norm, clip_grad_norm, clip_grad_value,
                        Top-K
  --max_norm MAX_NORM   gradient norm clipping max value (necessary if method:
                        clip_grad_norm)
  --top_k TOP_K         number of top elements to keep in compressed gradient
                        (necessary if method: Top-K)
  --clip_value CLIP_VALUE
                        gradient clipping value (necessary if method:
                        clip_grad_value)
```
> Set --beta 1 in case you need to train DistributedSGD instead of DistributedSGDM.
> As a result of running this code; data folder, FashionMNIST_ResNet18_CSV folder , and runs folder will be created.
> dataset will be downloaded in data folder.
> You can simultaneously monitor training progress through tensorboard by the files saved in runs folder.
> Training process will be loged also in a CSV file in FashionMNIST_ResNet18_CSV folder.

## How to Aggregate CSV Files and generate Mean and STD over different trials
> Use `aggregateCSVs .ipynb` to generate a single CSV file containing the mean and standard deviation of 5 runs on each experiment's setup.

## How to Plot the Results
> To draw output figures with the desired features use `PlotResults.ipynb`.


# Citation
* Submitted to the Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS 2023), New Orleans Ernest N. Morial Convention Center, USA, Dec 10 – Dec 16, 2023.

Please cite the accompanied paper, if you find this useful:
```
To be completed
```
