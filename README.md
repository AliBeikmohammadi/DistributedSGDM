# DistributedSGDM
Implementation of **DistributedSGDM**, as presented in:
* Distributed Momentum Methods Under Biased Gradients Estimation. Submitted to the IEEE Transactions on Neural Networks and Learning Systems.


# Additional Numerical Evaluations on Deep Neural Networks
Additional learning curves are included here. To be more specific, Figures 1 and  2  contain learning curves for the MNIST dataset using FCNN. Also, for the FashionMNIST dataset, more results are shown in Figures 3, 4, 5, 6, 7, and  8, utilizing ResNet-18 model. The shaded regions correspond to the standard deviation of the average evaluation over five trials.

1. (Figure 1) Results on MNIST dataset, considering $n=100$ and $\gamma=0.5$ - FCNN.

  
  * Top-5\% sparsification


![Top-5\% sparsification](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.5_beta_0.1_TopK_33500.png)


  * Clipped with $\tau=1$

  
![Clipped with $\tau=1$](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.5_beta_0.1_ClipNorm_1.0.png)


  * Top-10\% sparsification
 

![Top-10\% sparsification](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.5_beta_0.1_TopK_67000.png)


  * Clipped with $\tau=5$
 

![Clipped with $\tau=5$](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.5_beta_0.1_ClipNorm_5.0.png)


2. (Figure 2) Results on MNIST dataset, considering $n=100$ and $\gamma=0.7$ - FCNN.

  
  * Top-1\% sparsification


![Top-5\% sparsification](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.7_beta_0.1_TopK_13400.png)


  * Clipped with $\tau=1$

  
![Clipped with $\tau=1$](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.7_beta_0.1_ClipNorm_1.0.png)


  * Top-5\% sparsification
 

![Top-10\% sparsification](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.7_beta_0.1_TopK_33500.png)


  * Clipped with $\tau=2$
 

![Clipped with $\tau=5$](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.7_beta_0.1_ClipNorm_2.0.png)


  * Top-10\% sparsification
 

![Top-10\% sparsification](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.7_beta_0.1_TopK_67000.png)


  * Clipped with $\tau=5$
 

![Clipped with $\tau=5$](/MNISTRunPy/MNIST_Plots_Paper_WOT/MNIST_A_512512_penalty_0.0_e_200_N_100_SGDM_lr_0.7_beta_0.1_ClipNorm_5.0.png)

 
 3. (Figure 3) Results on FashionMNIST dataset, considering $n=100$ and $\gamma=0.3$ - ResNet-18.

  
  * Top-0.1\% sparsification


![Top-5\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.1_TopK_11181.png)


  * Clipped with $\tau=1$

  
![Clipped with $\tau=1$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.1_ClipNorm_1.0.png)


  * Top-1\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.1_TopK_111816.png)


  * Clipped with $\tau=2$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.1_ClipNorm_2.0.png)


  * Top-10\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.1_TopK_1118164.png)


  * Clipped with $\tau=5$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.1_ClipNorm_5.0.png)


 4. (Figure 4) Results on FashionMNIST dataset, considering $n=100$ and $\gamma=0.3$ - ResNet-18.

  
  * Top-0.1\% sparsification


![Top-5\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.3_TopK_11181.png)


  * Clipped with $\tau=1$

  
![Clipped with $\tau=1$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.3_ClipNorm_1.0.png)


  * Top-1\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.3_TopK_111816.png)


  * Clipped with $\tau=2$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.3_ClipNorm_2.0.png)


  * Top-10\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.3_TopK_1118164.png)


  * Clipped with $\tau=5$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.3_beta_0.3_ClipNorm_5.0.png)


 5. (Figure 5) Results on FashionMNIST dataset, considering $n=100$ and $\gamma=0.5$ - ResNet-18.


  * Top-1\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.1_TopK_111816.png)


  * Clipped with $\tau=1$

  
![Clipped with $\tau=1$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.1_ClipNorm_1.0.png)


  * Top-10\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.1_TopK_1118164.png)


  * Clipped with $\tau=5$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.1_ClipNorm_5.0.png)


 6. (Figure 6) Results on FashionMNIST dataset, considering $n=100$ and $\gamma=0.5$ - ResNet-18.

  
  * Top-0.1\% sparsification


![Top-5\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.3_TopK_11181.png)


  * Clipped with $\tau=1$

  
![Clipped with $\tau=1$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.3_ClipNorm_1.0.png)


  * Top-1\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.3_TopK_111816.png)


  * Clipped with $\tau=2$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.3_ClipNorm_2.0.png)


  * Top-10\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.3_TopK_1118164.png)


  * Clipped with $\tau=5$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.5_beta_0.3_ClipNorm_5.0.png)


 7. (Figure 7) Results on FashionMNIST dataset, considering $n=100$ and $\gamma=0.7$ - ResNet-18.

  
  * Top-0.1\% sparsification


![Top-5\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.1_TopK_11181.png)


  * Clipped with $\tau=1$

  
![Clipped with $\tau=1$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.1_ClipNorm_1.0.png)


  * Top-1\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.1_TopK_111816.png)


  * Clipped with $\tau=2$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.1_ClipNorm_2.0.png)


  * Top-10\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.1_TopK_1118164.png)


  * Clipped with $\tau=5$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.1_ClipNorm_5.0.png)


 8. (Figure 8) Results on FashionMNIST dataset, considering $n=100$ and $\gamma=0.7$ - ResNet-18.

  
  * Top-0.1\% sparsification


![Top-5\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.3_TopK_11181.png)


  * Clipped with $\tau=1$

  
![Clipped with $\tau=1$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.3_ClipNorm_1.0.png)


  * Top-1\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.3_TopK_111816.png)


  * Clipped with $\tau=2$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.3_ClipNorm_2.0.png)


  * Top-10\% sparsification
 

![Top-10\% sparsification](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.3_TopK_1118164.png)


  * Clipped with $\tau=5$
 

![Clipped with $\tau=5$](/FashionMNISTRunPy/FashionMNIST_ResNet18_Plots_Paper_WOT/FashionMNIST_A_ResNet18_penalty_0.0_e_300_N_100_SGDM_lr_0.7_beta_0.3_ClipNorm_5.0.png)


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
> 
> As a result of running this code; data folder, MNIST_CSV folder , and runs folder will be created.

* dataset will be downloaded in data folder.

* You can simultaneously monitor training progress through tensorboard by the files saved in runs folder.

* Training process will be loged also in a CSV file in MNIST_CSV folder.


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
> 
> As a result of running this code; data folder, FashionMNIST_ResNet18_CSV folder , and runs folder will be created.
* dataset will be downloaded in data folder.
* You can simultaneously monitor training progress through tensorboard by the files saved in runs folder.
* Training process will be loged also in a CSV file in FashionMNIST_ResNet18_CSV folder.

## How to Aggregate CSV Files and generate Mean and STD over different trials
> Use `aggregateCSVs.ipynb` to generate a single CSV file containing the mean and standard deviation of 5 runs on each experiment's setup.

## How to Plot the Results
> To draw output figures with the desired features use `PlotResults.ipynb`.


# Citation
* Submitted to the IEEE Transactions on Neural Networks and Learning Systems.

Please cite the accompanied paper, if you find this useful:
```
To be completed
```
