# :gem: GEM: Generalization-Efficient Methods for image classification with small datasets

**GEM** is a PyTorch-based library with the goal of providing a shared codebase for fast prototyping, training and reproducible evaluation of learning algorithms that generalize on small image datasets.

In particular, the repository contains all the tools to reproduce and possibly extend the experiments of the paper [Image Classification with Small Datasets: Overview and Benchmark](https://ieeexplore.ieee.org/abstract/document/9770050). It provides:

- [x] A (possibly extendable) benchmark of 5 datasets spanning various data domains and types
- [x] A realistic and fair experimental pipeline including hyper-parameter optimization and common training set-ups
- [x] A (possibly extendable) large pool of implementations for state-of-the-art methods  

Given the "living" nature of our libary, we plan in the future to introduce and keep the repository updated with new approaches and datasets to drive further progress toward small-sample learning methods.

## :bookmark_tabs: Table of Contents   
- [Overview](#book-overview)<a name="book-overview"/>
   - [Structure](#structure)<a name="structure"/>
   - [Datasets](#datasets)<a name="datasets"/>
   - [Methods](#methods)<a name="methods"/>
- [Usage](#gear-usage)<a name="gear-usage"/>
   - [Installation](#installation)<a name="installation"/>
   - [Method Evaluation](#method-evaluation)<a name="method-evaluation"/>
   - [Library Extension](#library-extension)<a name="library-extension"/>
- [Results](#bar_chart-results)<a name="bar_chart-results"/>
- [Citation](#writing_hand-citation)<a name="writing_hand-citation"/>




## :book: Overview

### Structure

More details soon!

### Datasets

The datasets constituting our benchmark are the following:

|      Dataset      | Classes | Imgs/Class | Trainval |  Test  | Problem Domain |   Data Type   |   Identifier   |
|:------------------|--------:|-----------:|---------:|-------:|:---------------|:--------------|:---------------|
| [ciFAIR-10][1] \* |      10 |         50 |      500 | 10,000 | Natural Images | RGB (32x32)   | ```cifair10``` |
| [CUB][2]          |     200 |         30 |    5,994 |  5,794 | Fine-Grained   | RGB           | ```cub```      |
| [ISIC 2018][3] \* |       7 |         80 |      560 |  1,944 | Medical        | RGB           | ```isic2018``` |
| [EuroSAT][4] \*   |      10 |         50 |      500 | 19,500 | Remote Sensing | Multispectral | ```eurosat```  |
| [CLaMM][5] \*     |      12 |         50 |      600 |  2,000 | Handwriting    | Grayscale     | ```clamm```    |

\* We use subsampled versions of the original datasets with fewer images per class.

For additional details on the dataset statistics, splits, and ways to download the data, visit the respective page in the folder [datasets](datasets).
The directory contains one sub-directory for each dataset in our benchmark. These directories contain the split files specifying the subsets of data employed in our experiments. The files ```trainval{i}.txt``` are simply the concatenation of ```train{i}.txt``` and ```val{i}.txt``` (with ```i``` in {0,1,2}). These subsets can be used for the final training before evaluating a method on the test set. Development and hyper-parameter optimization (HPO), however, should only be conducted using the training and validation sets.

The aforementioned files list all images contained in the respective subset, one per line, along with their class labels. Each line contains the filename of an image followed by a space and the numeric index of its label.

The only exception from this common format is [ciFAIR-10](datasets/cifair), since it does not have filenames. A description of the split can be found in the README of the respective directory.

### Methods

We currently provide the implementations of the following methods:

|                      Method                      | Original code            |  Our implementation                             | Identifier         |
|:-------------------------------------------------|-------------------------:|------------------------------------------------:|-------------------:|
| Cross-Entropy Loss (baseline)                    |      --                  |[```xent.py```][xent.py]                         |```xent```          |
| [Deep Hybrid Networks][scattering]               |[link][scattering_code]   |[```scattering.py```][scattering.py]             |```scattering```    |
| [OLÉ][ole]                                       |[link][ole_code]          |[```ole.py```][ole.py]                           |```ole```           |
| [Grad-L2 Penalty][kernelregular]                 |[link][kernelregular_code]|[```kernelregular.py```][kernelregular.py]       |```gradl2```        |
| [Cosine Loss (+ Cross-Entropy)][cosineloss]      |--                        |[```cosineloss.py```][cosineloss.py]             |```cosine```        |
| [Harmonic Networks][harmonic]                    |[link][harmonic_code]     |[```harmonic.py```][harmonic.py]                 |```harmonic```      |
| [Full Convolution][fconv]                        |[link][fconv_code]        |[```fconv.py```][fconv.py]                       |```fconv```         |
| [DSK Networks][dsknet]                           |--                        |[```dsk_classifier.py```][dsk_classifier.py]     |       ```dsk```    |
| [Distilling Visual Priors][distill]              |[link][distill_code]      |[```distill_pretraining.py```][distill_pretraining.py]<br>[```distill_classifier.py```][distill_classifier.py]|  ```dvp-pretrain```<br>```dvp-distill```|
| [Auxiliary Learning][auxilearn]                  |[link][auxilearn_code]    |   [```auxilearn.py```][auxilearn.py]            |```auxilearn```     |
| [T-vMF Similarity][tvmf]                         |[link][tvmf_code]         |   [```tvmf.py```][tvmf.py]                      |```tvmf```          |


## :gear: Usage

### Installation

To use the repository, clone it in your local system:

```
git clone https://github.com/lorenzobrigato/gem.git
```

and install the required packages with:

```
python -m pip install -r requirements.txt
```

**Note**: GEM requires PyTorch with GPU support. Hence, for instructions on how to install PyTorch versions compatible with your CUDA versions, see [pytorch.org](pytorch.org).


### Method Evaluation

We provide a set of scripts located in the directories [scripts](scripts) and [bash_scripts](bash_scripts) to reproduce the experimental pipeline presented in our paper. In particular, evaluating one method on the full benchmark consists in:

1. Finding hyper-parameters by training the approach on the ```train{i}.txt``` split while evaluating on the respective ```val{i}.txt```
2. Training 10 instances of the method given the found configuration on the full ```trainval{i}.txt``` split while evaluating on the test split
3. Repeating independently points 1. and 2. for all values of ```i```

For all datasets, the number of training splits used in our paper is 3, hence ```i``` is in the range {0,1,2}. For the testing sets, in some cases we have multiple splits as for the training, in others we employed a single ```test0.txt``` split. We performed multiple independent evaluations changing dataset splits and optimization runs to account for random variance (particularly significant in the small-sample regime).

To separately perform 1. and 2., we respectively provide [```hpo.py```](scripts/hpo.py) and [```train.py```](scripts/train.py) / [```train_ray.py```](scripts/train_ray.py). It is also possible to do 1. and 2. sequentially by executing [```full_train.py```](scripts/full_train.py).
For achieving 3., refer to the bash scripts available in [bash_scripts](bash_scripts). 
We are now going to treat in more details all the available chioces in terms of runnable scripts.

#### Hyper-Parameter Optimization (HPO)

For what concerns HPO, we employ an efficient and easy-to-use library ([Tune](https://docs.ray.io/en/latest/tune/index.html#)) and a state-of-the-art search algorithm ([Asynchronous Successive Halving
Algorithm (ASHA)](https://proceedings.mlsys.org/paper/2020/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)).

Script [```hpo.py```](scripts/hpo.py) is dedicated to finding hyper-parameters of a method.
For instance, searching for default hyper-parameters, i.e., learning rate, weight decay, and batch size, for the cross-entropy baseline with a Wide ResNet-16-8 on the [ciFAIR-10][1] dataset and splits 0 (default) is achievable by running:

```bash
python scripts/hpo.py cifair10 \
--method xent \
--architecture wrn-16-8 \
--rand-shift 4 \
--epochs 500 \
--grace-period 50 \
--num-trials 250 \
--cpus-per-trial 8 \
--gpus-per-trial 0.5
```
After completion, the script will print on screen the found hyper-parameters. Notice that ```--grace-period``` and ```--num-trials``` refer to parameters of the search algorithm. that have been fixed for each dataset and are hard-coded in the bash scripts of folder [bash_scripts](bash_scripts).
To have a complete view of all the arguments accepted by the script, chek the help message of the parser by running: 

```
python scripts/hpo.py -h
```
Note also that you can configure the hardware resources spent on trials. For examle, with ```--gpus-per-trial 0.5``` the script will run two trials in parallel.
Exploit parallelism to speed up the search but consider that the number of trials per GPU is bounded by the GPU memory available. 

#### Final Evaluation

Once that the hyper-parameters have been found, you can execute the training of a single model for the test evaluation with script [```train.py```](scripts/train.py). Or you can also train multiple instances of the same model in parallel exploiting again the [Tune](https://docs.ray.io/en/latest/tune/index.html#) library and script [```train_ray.py```](scripts/train_ray.py).

An example to train 10 instances of the baseline method with possibly found hyper-parameters (learning rate, weight decay, and batch size) is:

```bash
python scripts/train_ray.py cifair10 \
--method xent \
--architecture wrn-16-8 \
--rand-shift 4 \
--epochs 500 \
--lr 4.55e-3 \
--weight-decay 5.29e-3 \
--batch-size 10 \
--num-trials 10 \
--cpus-per-trial 8 \
--gpus-per-trial 0.5 \
--eval-interval 10 \
--save /home/user/gem_models/cifair10/cifair10_xent.pth \
--history /home/user/gem_logs/cifair10/cifair10_xent.json
```
Note that we are saving the model file and the history log containing the results by specifying the ```--save``` and ```history``` arguments.

#### Full Training

The HPO and final evaluation steps can be executed sequentially and from the same script [```full_train.py```](scripts/full_train.py). Most of the arguments are shared with the previous scripts. A key difference regards the pattern "-f" that is added at the end of some arguments with the objective of discerning the two training phases. E.g, given ```--num-trials 250``` and ```--num-trials-f 10```, the script will run 250 trials for hyper-parameter optimization, and 10 trials for the final evaluation.
For additional details refer to the help message of the parser:

```
python scripts/full_train.py -h
```

#### Multi-Split Training

To obtain a complete evaluation on one of the datasets of our benchmark is necessary to repeat the full training on the 3 splits.
This is achievable by running one of the bash scripts in [bash_scripts](bash_scripts). Each of those scripts sequentially runs [```full_train.py```](scripts/full_train.py).

**Note**: the default configurations for dataset-specific augmentations and parameters of the search algorithm are hard-coded inside the scripts. Any additional argument needed for the full training can be added in the call of the bash script. An example for the baseline training on [ciFAIR-10](datasets/cifair) is:

```bash
bash bash_scripts/bench_cifair10.sh \
--method xent \
--cpus-per-trial 8 \
--gpus-per-trial 0.5 \
--eval-interval-f 10 \
--save-f /home/user/gem_models/cifair10/cifair10_xent.pth \
--history-f /home/user/gem_logs/cifair10/cifair10_xent.json
```
Given that multiple models/logs are saved, [```full_train.py```](scripts/full_train.py) also adds a temporal pattern representing the unique timestamp rigth before the file extensions of the names provided at ```--save-f``` / ```--history-f```.

#### Evaluation

Script [```evaluate_baccuracy_json.py```](scripts/evaluate_baccuracy_json.py) is available to compute balanced accuracy from single runs and mean/standard deviation over multiple runs. It also eventually save in a more compact format (JSON) a summary of such results.
For more info execute:

```
python scripts/evaluate_baccuracy_json.py -h
```

### Library Extension

More details soon!


## :bar_chart: Results

Here are the full results for all methods currently evaluated on our benchmark:

|   Method                                  | [ciFAIR-10][1] |  [CUB][2]  | [ISIC 2018][3] | [EuroSAT][4] | [CLaMM][5] |    Avg.    |
|:------------------------------------------|---------------:|-----------:|---------------:|-------------:|-----------:|-----------:|
| Cross-Entropy Loss (baseline)             |       55.18%   |   70.79%   |       64.49%   |     90.58%   |   70.15%   |   70.24%   |
| [Deep Hybrid Networks][scattering]        |       53.84%   |   55.37%   |       62.06%   |     88.77%   |   63.75%   |   64.76%   |
| [OLÉ][ole]                                |       55.19%   |   66.55%   |       62.80%   |     90.29%   |   74.28%   |   69.82%   |
| [Grad-L2 Penalty][kernelregular]          |       51.90%   |   51.94%   |       60.21%   |     81.50%   |   65.10%   |   62.13%   |
| [Cosine Loss][cosineloss]                 |       52.39%   |   66.94%   |       62.42%   |     88.53%   |   68.89%   |   67.83%   |
| [Cosine Loss + Cross-Entropy][cosineloss] |       52.77%   |   70.43%   |       63.17%   |     89.65%   |   70.64%   |   69.33%   |
| [Harmonic Networks][harmonic]             |     **58.00%** | **73.07%** |     **69.70%** |   **91.98%** | **77.25%** | **74.00%** |
| [Full Convolution][fconv]                 |       54.64%   |   63.74%   |       57.34%   |     89.47%   |   69.06%   |   66.85%   |
| [DSK Networks][dsknet]                    |       53.84%   |   69.75%   |       63.41%   |     91.09%   |   65.43%   |   68.70%   |
| [Distilling Visual Priors][distill]       |      *57.80%*  |   70.81%   |       62.39%   |     88.96%   |   69.07%   |   69.81%   |
| [Auxiliary Learning][auxilearn]           |       51.84%   |   43.57%   |       61.70%   |     80.92%   |   60.24%   |   59.65%   |
| [T-vMF Similarity][tvmf]                  |       56.75%   |   68.19%   |       64.60%   |     88.50%   |   69.33%   |   69.47%   |

All values represent the balanced classification accuracy averaged over 30 training runs. Precisely, three groups of 10 runs over the three dataset splits.  
**Bold** results are the best in their column and *italic* results are not significantly worse than the best (on a level of 5%).

CUB, ISIC, and CLaMM have unbalanced test sets.
For the other datasets, balanced classification accuracy is equivalent to standard accuracy.

## :writing_hand: Citation

If you find this repository useful to your research, please consider citing our paper

```
@ARTICLE{9770050,
author={Brigato, Lorenzo and Barz, Björn and Iocchi, Luca and Denzler, Joachim},
journal={IEEE Access},
title={Image Classification With Small Datasets: Overview and Benchmark},
year={2022},
volume={10},
pages={49233-49250},
doi={10.1109/ACCESS.2022.3172939}
}
```

[1]: datasets/cifair
[2]: datasets/cub
[3]: datasets/isic2018
[4]: datasets/eurosat
[5]: datasets/clamm
[cosineloss]: https://arxiv.org/abs/1901.09054
[harmonic]: https://arxiv.org/abs/1905.00135
[kernelregular]: https://arxiv.org/abs/1810.00363
[scattering]: https://arxiv.org/abs/1703.08961
[distill]: https://arxiv.org/abs/2008.00261
[dsknet]: https://link.springer.com/chapter/10.1007%2F978-3-030-66096-3_35
[fconv]: https://arxiv.org/abs/2003.07064
[ole]: https://arxiv.org/abs/1712.01727
[auxilearn]: https://arxiv.org/abs/2007.02693
[tvmf]: https://openaccess.thecvf.com/content/CVPR2021/html/Kobayashi_T-vMF_Similarity_for_Regularizing_Intra-Class_Feature_Distribution_CVPR_2021_paper.html

[xent.py]: gem/pipelines/xent.py
[cosineloss.py]: gem/pipelines/cosineloss.py
[harmonic.py]: gem/pipelines/harmonic.py
[kernelregular.py]: gem/pipelines/kernelregular.py
[scattering.py]: gem/pipelines/scattering.py
[distill_pretraining.py]: gem/pipelines/distill_visual_priors/distill_pretraining.py
[distill_classifier.py]: gem/pipelines/distill_visual_priors/distill_classifier.py
[dsk_classifier.py]: gem/pipelines/dsk/dsk_classifier.py
[fconv.py]: gem/pipelines/fconv.py
[ole.py]: gem/pipelines/ole.py
[auxilearn.py]: gem/pipelines/auxilearn/auxilearn_classifier.py
[tvmf.py]: gem/pipelines/tvmf.py

[harmonic_code]: https://github.com/matej-ulicny/harmonic-networks
[kernelregular_code]: https://github.com/albietz/kernel_reg
[scattering_code]: https://github.com/edouardoyallon/scalingscattering
[distill_code]: https://github.com/DTennant/distill_visual_priors
[fconv_code]: https://github.com/oskyhn/CNNs-Without-Borders
[ole_code]: https://github.com/jlezama/OrthogonalLowrankEmbedding
[auxilearn_code]: https://github.com/AvivNavon/AuxiLearn
[tvmf_code]: https://github.com/tk1980/tvMF
