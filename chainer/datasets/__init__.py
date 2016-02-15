from chainer.datasets.adapter import cross_validation
from chainer.datasets.adapter import multiprocess_loader
from chainer.datasets.adapter import sub_dataset
from chainer.datasets.example import mnist
from chainer.datasets.example import ptb_words
from chainer.datasets import image_dataset
from chainer.datasets import simple_dataset

CrossValidationTrainingDataset = \
    cross_validation.CrossValidationTrainingDataset
CrossValidationTestDataset = cross_validation.CrossValidationTestDataset
get_cross_validation_datasets = cross_validation.get_cross_validation_datasets
MultiprocessLoader = multiprocess_loader.MultiprocessLoader
SubDataset = sub_dataset.SubDataset

MnistTraining = mnist.MnistTraining
MnistTest = mnist.MnistTest
PTBWordsTraining = ptb_words.PTBWordsTraining
PTBWordsValidation = ptb_words.PTBWordsValidation
PTBWordsTest = ptb_words.PTBWordsTest

ImageDataset = image_dataset.ImageDataset
ImageListDataset = image_dataset.ImageListDataset
SimpleDataset = simple_dataset.SimpleDataset
