# Caltech-UCSD Birds-200-2011 Dataset

CUB is a fine-grained visual recognition dataset comprising 11,788 images of 200 different bird species.

Dataset homepage: <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>  
Paper: <https://authors.library.caltech.edu/27452/1/CUB_200_2011.pdf>

![Example images from CUB](http://www.vision.caltech.edu/visipedia/collage.jpg)


## Obtaining the data

The bash script `download_data.sh` provided in this directory can be used to download the actual image data. Doing so will result in a folder named `CUB_200_2011` with a subdirectory called `images`.


## Splits

We provide the following splits of the dataset for testing small-data performance:

|   Split      | Total Images | Images / Class |
|:-------------|-------------:|---------------:|
| train{i}     |        4,000 |             20 |
| val{i}       |        1,994 |           9-10 |
| trainval{i}  |        5,994 |          29-30 |
| test0        |        5,794 |          11-30 |

To be consistent with the other datasets, we name the training splits as `trainval{i}` despite they coincide. The original CUB split is too small to allow different splits. The split `test0` also corresponds to the original `test`.
We subdivide `trainval{i}` into `train{i}` and `val{i}`.
The value of `i` ranges in {0,1,2}.


## Usage

This dataset can be loaded using `gem.datasets.CUBDataset`.
The dataset identifier is `"cub"`.
