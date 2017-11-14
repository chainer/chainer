# Large Scale ConvNets

## Requirements

- Pillow (Pillow requires an external library that corresponds to the image format)

## Description

This is an experimental example of learning from the ILSVRC2012 classification dataset.
It requires the training and validation dataset of following format:

* Each line contains one training example.
* Each line consists of two elements separated by space(s).
* The first element is a path to 256x256 RGB image.
* The second element is its ground truth label from 0 to 999.

The text format is equivalent to what Caffe uses for ImageDataLayer.
This example currently does not include dataset preparation script.

This example requires "mean file" which is computed by `compute_mean.py`.


# Additional Instructions to run out-of-core training

## Installation

Followings are instructions to install chainer out-of-core training code.
Basically you can install it by [official chainer instructions from source code](https://docs.chainer.org/en/stable/install.html#install-chainer-from-source). These instructions are additional instructions of the official instructions.

Chainer out-of-core training code depends on a branch of cupy. So, please use following branch.
```sh
$ git clone https://github.com/imaihal/cupy.git
$ cd cupy
$ git checkout -b v2-trl-swapinout origin/v2-trl-swapinout
$ python setup.py install
```

Then, please use a branch for Chainer out-of-core
```sh
$ git clone https://github.com/imaihal/chainer.git
$ cd chainer
$ git checkout -b v3-trl-ooc-pr origin/v3-trl-ooc-pr
$ python setup.py install
```

## Prepare enlarged dataset of Imagenet
You can run chainer out-of-core using original imagenet dataset (256x256), but it is effective to use larger dataset in out-of-core training. Following command converts the original datatset to 10 times larger dataset(2560x2560). You can create mean file by `compute_mean.py` using these enlarged dataset.

```sh
$ convert <infile> -resize 2560x2560! <outfile>
```

## Run

You can use new argument of "--insize" and "--ooc". The "--insize" specifies input data size and the "--ooc" enables the out-of-core training. Currently, the "--insize" is supported in googlenet`(googlenet.py)` and ResNet50 `(resnet50.py)`. You can specify the size of multiple of 224 in the "--insize".
Following is an example to run googlenet with insize 2240x2240 on 4GPUs.
```sh
$ python -u train_imagenet_data_parallel_OOC_ibm.py --arch googlenet --insize 2240 --batchsize <mini batch size> --iteration <iterations> --gpu 0 1 2 3 --mean <mean file> --loaderjob <num of parallel data loading processes> --val_batchsize <Validation minibatch size> --out <Output directory> --ooc --root <Root directory path of image files> <Path to training image-label list file> <Path to validation image-label list file>
```

## Reference

2 PRs are created for chainer out-of-core training
* [\[WIP\] Out-of-core training on V3 #3762](https://github.com/chainer/chainer/pull/3762)
* [\[WIP\] Swap in/out between GPU and CPU memory on v2 #694](https://github.com/cupy/cupy/pull/694)
