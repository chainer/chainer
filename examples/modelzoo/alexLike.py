import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

class AlexLike(chainer.Chain):
    insize = 128
    def __init__(self, n_out):
        super(AlexLike, self).__init__(
            conv1=L.Convolution2D(None,  64, 8, stride=4),
            conv2=L.Convolution2D(None, 128,  5, pad=2),
            conv3=L.Convolution2D(None, 128,  3, pad=1),
            conv4=L.Convolution2D(None, 128,  3, pad=1),
            conv5=L.Convolution2D(None, 64,  3, pad=1),
            fc6=L.Linear(None, 1024),
            fc7=L.Linear(None, 1024),
            fc8=L.Linear(None, n_out),
        )
        self.train = True

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.fc7(h)), train=self.train)
        h = self.fc8(h)
        return h

class FromCaffeAlexnet(chainer.Chain):
    insize = 128
    def __init__(self, n_out):
        super(FromCaffeAlexnet, self).__init__(
            conv1=L.Convolution2D(None, 96, 11, stride=2),
            conv2=L.Convolution2D(None, 256, 5, pad=2),
            conv3=L.Convolution2D(None, 384, 3, pad=1),
            conv4=L.Convolution2D(None, 384, 3, pad=1),
            conv5=L.Convolution2D(None, 256, 3, pad=1),
            my_fc6=L.Linear(None, 4096),
            my_fc7=L.Linear(None, 1024),
            my_fc8=L.Linear(None, n_out),
        )
        self.train = True

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.my_fc7(h)), train=self.train)
        h = self.my_fc8(h)
        return h
