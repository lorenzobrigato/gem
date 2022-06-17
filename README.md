# :gem: GEM: Generalization-Efficient Methods for image classification with small datasets

**GEM** is a Pytorch-based library with the goal of providing a shared codebase for fast prototyping, training and reproducible evaluation of learning algorithms that generalize on small image datasets.

In particular, the repository contains all the tools to reproduce and possibly extend the experiments of the paper [Image Classification with Small Datasets: Overview and Benchmark](https://ieeexplore.ieee.org/abstract/document/9770050). It provides:

- [x] A (possibly extendable) benchmark of 5 datasets spanning various data domains and types
- [x] A realistic and fair experimental pipeline including hyper-parameter optimization (HPO) and common training set-ups
- [x] A (possibly extendable) large pool of implementations for state-of-the-art methods  

Given the "living" nature of our libary, we plan in the future to introduce and keep the repository updated with new approaches and datasets to drive further progress toward small-sample learning methods.

## :book: Overview

### Structure

More details soon!

### Datasets

The datasets constituting our benchmark are the following:

|      Dataset      | Classes | Imgs/Class | Trainval |  Test  | Problem Domain |   Data Type   |
|:------------------|--------:|-----------:|---------:|-------:|:---------------|:--------------|
| [ciFAIR-10][1] \* |      10 |         50 |      500 | 10,000 | Natural Images | RGB (32x32)   |
| [CUB][2]          |     200 |         30 |    5,994 |  5,794 | Fine-Grained   | RGB           |
| [ISIC 2018][3] \* |       7 |         80 |      560 |  1,944 | Medical        | RGB           |
| [EuroSAT][4] \*   |      10 |         50 |      500 | 19,500 | Remote Sensing | Multispectral |
| [CLaMM][5] \*     |      12 |         50 |      600 |  2,000 | Handwriting    | Grayscale     |

\* We use subsampled versions of the original datasets with fewer images per class.

For additional details on the dataset statistics, splits, and ways to download the data, visit the respective page in the folder [datasets](./datasets).
The directory contains one sub-directory for each dataset in our benchmark. These directories contain the split files specifying the subsets of data employed in our experiments. The files ```trainval{i}.txt``` are simply the concatenation of ```train{i}.txt``` and ```val{i}.txt``` (with ```i``` in {0,1,2}). These subsets can be used for the final training before evaluating a method on the test set. Development and hyper-parameter optimization (HPO), however, should only be conducted using the training and validation sets.

The aforementioned files list all images contained in the respective subset, one per line, along with their class labels. Each line contains the filename of an image followed by a space and the numeric index of its label.

The only exception from this common format is [ciFAIR-10](./datasets/cifair), since it does not have filenames. A description of the split can be found in the README of the respective directory.

### Methods

We currently provide the implementations of the following methods:

|                      Method                      | Original code            | Identifier         |  Our implementation                             | 
|:-------------------------------------------------|-------------------------:|-------------------:|------------------------------------------------:|
| Cross-Entropy Loss (baseline)                    |      --                  |```xent```          |[```xent.py```][xent.py]                         |
| [Deep Hybrid Networks][scattering]               |[link][scattering_code]   |```scattering```    |[```scattering.py```][scattering.py]             |
| [OLÉ][ole]                                       |[link][ole_code]          |```ole```           |[```ole.py```][ole.py]                           |
| [Grad-L2 Penalty][kernelregular]                 |[link][kernelregular_code]|```gradl2```        |[```kernelregular.py```][kernelregular.py]       |
| [Cosine Loss (+ Cross-Entropy)][cosineloss]      |--                        |```cosine```        |[```cosineloss.py```][cosineloss.py]             |
| [Harmonic Networks][harmonic]                    |[link][harmonic_code]     |```harmonic```      |[```harmonic.py```][harmonic.py]                 |
| [Full Convolution][fconv]                        |[link][fconv_code]        |```fconv```         |[```fconv.py```][fconv.py]                       |
| [DSK Networks][dsknet]                           |--                        |       ```dsk```    |[```dsk_classifier.py```][dsk_classifier.py]     |
| [Distilling Visual Priors][distill]              |[link][distill_code]      |```dvp-pretrain```<br>```dvp-distill```|[```distill_pretraining.py```][distill_pretraining.py]<br>[```distill_classifier.py```][distill_classifier.py]|  
| [Auxiliary Learning][auxilearn]                  |[link][auxilearn_code]    |```auxilearn```     |   [```auxilearn.py```][auxilearn.py]            |
| [T-vMF Similarity][tvmf]                         |[link][tvmf_code]         |```tvmf```          |   [```tvmf.py```][tvmf.py]                      |


## :gear: Usage

Refer to the Python and bash scripts located in the directories [scripts](./scripts) and [bash_scripts](./bash_scripts).
More details about usage are going to be available soon!

## :bar_chart: Results

Here are the full results for all methods evaluated on our benchmark:

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

[xent.py]: ./gem/pipelines/xent.py
[cosineloss.py]: ./gem/pipelines/cosineloss.py
[harmonic.py]: ./gem/pipelines/harmonic.py
[kernelregular.py]: ./gem/pipelines/kernelregular.py
[scattering.py]: ./gem/pipelines/scattering.py
[distill_pretraining.py]: ./gem/pipelines/distill_visual_priors/distill_pretraining.py
[distill_classifier.py]: ./gem/pipelines/distill_visual_priors/distill_classifier.py
[dsk_classifier.py]: ./gem/pipelines/dsk/dsk_classifier.py
[fconv.py]: ./gem/pipelines/fconv.py
[ole.py]: ./gem/pipelines/ole.py
[auxilearn.py]: ./gem/pipelines/auxilearn/auxilearn_classifier.py
[tvmf.py]: ./gem/pipelines/tvmf.py

[harmonic_code]: https://github.com/matej-ulicny/harmonic-networks
[kernelregular_code]: https://github.com/albietz/kernel_reg
[scattering_code]: https://github.com/edouardoyallon/scalingscattering
[distill_code]: https://github.com/DTennant/distill_visual_priors
[fconv_code]: https://github.com/oskyhn/CNNs-Without-Borders
[ole_code]: https://github.com/jlezama/OrthogonalLowrankEmbedding
[auxilearn_code]: https://github.com/AvivNavon/AuxiLearn
[tvmf_code]: https://github.com/tk1980/tvMF
