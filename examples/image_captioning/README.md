# Image Captioning with Convolutional Neural Networks

This is an example implementation of Show and Tell: A Neural Image Caption Generator (https://arxiv.org/abs/1411.4555) a generative image captioning model using a neural network with convolutional and recurrent layers.
Given an image, this model generates a sentence that describes it.


## Requirements

This example requires the following packages.

- PIL
- [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI)

To install pycocotools, clone the repository and run `pip install -e .` from `cocoapi/PythonAPI` where `setup.py` is located.

## Model Overview

The model takes an image as input which is fed through a pretrained VGG16 model in order to extract features.
These features are then passed to a language model, a recurrent neural network that generates a caption word-by-word until the `EOS` (end-of-sentence) token is encountered or the caption reaches a maximum length.
During training, the loss is the softmax cross entropy of predicting the next word given the preceding words in the caption.

### More About the Language Model

The internals of the language models is a neural network with [LSTM](https://docs.chainer.org/en/stable/reference/generated/chainer.links.LSTM.html) layers.
However, Chainer also has a [NStepLSTM](https://docs.chainer.org/en/stable/reference/generated/chainer.links.NStepLSTM.html) layer which does not require sequential passes (for-loops in the code) which is faster. Using the latter, you do not have to align the caption lengths in the training data neither, which you usually do if using the former.
This example uses NStepLSTM by default, but also includes the equivalent code implmenented using standard LSTM as a reference.
When training with LSTM, you may want to specify the maximum caption length `--max-caption-length` to which all captions will be capped.

## Dataset

Run the following command to download the MSCOCO captioning dataset for training this model.

```bash
$ python download.py
```

This downloads and extracts the training and validation images, as well as necessary meta data including captions to a `data` directory under the current folder.
You can change the output directory by appeding `--out` followed by the target directory.
Notice that this may take a while and that it requires approximately 20 GB of disk space.

## Training

Once `download.py` finishes, you can start training the model.

```bash
$ python train.py --rnn nsteplstm --snapshot-iter 1000 --max-iters 50000 --batch-size 128 --gpu 0
```

If you run this script on Linux, setting the environmental variable `MPLBACKEND` to `Agg` may be required to use `matplotlib`. For example,

```
MPLBACKEND=Agg python train.py ...
```

The above example starts the training with the NStepLSTM layers in the language model and saves a snapshot of the trained model every 1000 iteration.
By default, the first model snapshot is saved as `result/model_1000`.

If you have specified a download directory for MSCOCO when preparing the dataset, add the `--mscoco-root` option followed by the path to that directory.

## Testing

To generate captions for new images, you need to have a snapshot of a trained model.
Assuming we are using the model snapshots from the training example after 20000 iterations, we can generate new captions as follows.

```bash
$ python predict.py --img cat.jpg --model result/model_20000 --rnn nsteplstm --max-caption-length 30 --gpu 0
```

It will print out the generated captions to std out.
If you want to generate captions to all images in a directory, replace `--img` with `--img-dir` followed by the directory.
Note that `--rnn` needs to given the correct value corresponding to the model.
