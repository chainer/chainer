#!/usr/bin/env python
# coding:utf-8

import os
import cPickle as pickle
from alexLike import *
from copy_model import *
from chainer import cuda, optimizers, Variable
from data_loader import *

DATA_ROOT_DIR_PATH    = "/mnt/nas101/hiroki11x/ILSVRC2012_img_val_256x256"
TESTING_DATA_PATH     = "/mnt/nas101/hiroki11x/val.txt"
MEAN_IMAGE_PATH       = "ilsvrc_2012_mean.npy"
FINE_TUNED_MODEL_PATH = "alexnet.pkl"

if __name__ == "__main__":
    # load a fine-tuned model
    model = pickle.load(open(FINE_TUNED_MODEL_PATH))

    # create a predictor
    predictor = AlexLike().to_gpu()

    # copy parameters from the fine-tuned model to the predictor
    copy_model(model, predictor)

    # load testing data
    data_loader = DataLoader()
    x_test, y_test = data_loader.load_with_subtraction_of_mean(
        TESTING_DATA_PATH,
        MEAN_IMAGE_PATH,
        AlexLike.insize
    )

    # reduce the data
    n = 20
    x = x_test[:n]
    y = y_test[:n]
    print(x.shape, y.shape)
    cx = Variable(cuda.to_gpu(x))
    cy = Variable(cuda.to_gpu(y))

    # predict
    predictor.train = False
    z = predictor(cx, cy)
    print(z.data.shape) # should display (n, 15)

    # compare predicted values with answers
    print("predicted values: {n}".format(n=np.argmax(z.data, axis=1)))
    print("answers:          {n}".format(n=y))
