# Evaluate a Caffe reference model

## Requirements

- Caffe model support (Python 2.7+, Protocol Buffers)
- Pillow (Pillow requires an external library that corresponds to the image format)

## Description

This is an example of evaluating a Caffe reference model using ILSVRC2012 classification dataset.
It requires the validation dataset in the same format as that for the imagenet example.

Model files can be downloaded by `download_model.py`. AlexNet and reference CaffeNet requires a mean file, which can be downloaded by `download_mean_file.py`.

## How to Run `evaluate_caffe_net.py`

```py
python evaluate_caffe_net.py /mnt/nas101/hiroki11x/val.txt --basepath /mnt/nas101/hiroki11x/ILSVRC2012_img_val_256x256 --batchsize 1 --mean ilsvrc_2012_mean.npy --gpu 0 alexnet bvlc_alexnet.caffemodel
```
