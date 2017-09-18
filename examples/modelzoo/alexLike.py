import numpy as np

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L

class AlexLike(chainer.Chain):
    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        super(AlexLike, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  96, 11, stride=4)
            self.conv2 = L.Convolution2D(None, 256,  5, pad=2)
            self.conv3 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(None, 384,  3, pad=1)
            self.conv5 = L.Convolution2D(None, 256,  3, pad=1)
            self.fc6 = L.Linear(None, 4096)
            self.fc7 = L.Linear(None, 4096)
            self.fc8 = L.Linear(None, 1000)

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        h = self.fc8(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)

        if self.train:
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred

    def clear(self):
        self.loss = None
        self.accuracy = None


class AlexLikeFp16(AlexLike):

    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 227

    def __init__(self):
        chainer.Chain.__init__(self)
        self.dtype = np.float16
        W = initializers.HeNormal(1 / np.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 96, 11, stride=4,
                                         initialW=W, initial_bias=bias)
            self.conv2 = L.Convolution2D(None, 256, 5, pad=2,
                                         initialW=W, initial_bias=bias)
            self.conv3 = L.Convolution2D(None, 384, 3, pad=1,
                                         initialW=W, initial_bias=bias)
            self.conv4 = L.Convolution2D(None, 384, 3, pad=1,
                                         initialW=W, initial_bias=bias)
            self.conv5 = L.Convolution2D(None, 256, 3, pad=1,
                                         initialW=W, initial_bias=bias)
            self.fc6 = L.Linear(None, 4096, initialW=W, initial_bias=bias)
            self.fc7 = L.Linear(None, 4096, initialW=W, initial_bias=bias)
            self.fc8 = L.Linear(None, 1000, initialW=W, initial_bias=bias)

    def __call__(self, x, t):
        return AlexLike.__call__(self, F.cast(x, self.dtype), t)

    def clear(self):
        self.loss = None
        self.accuracy = None

# class FromCaffeAlexnet(chainer.Chain):
#     insize = 128
#     def __init__(self, n_out):
#         super(FromCaffeAlexnet, self).__init__(
#             conv1=L.Convolution2D(None, 96, 11, stride=2),
#             conv2=L.Convolution2D(None, 256, 5, pad=2),
#             conv3=L.Convolution2D(None, 384, 3, pad=1),
#             conv4=L.Convolution2D(None, 384, 3, pad=1),
#             conv5=L.Convolution2D(None, 256, 3, pad=1),
#             my_fc6=L.Linear(None, 4096),
#             my_fc7=L.Linear(None, 1024),
#             my_fc8=L.Linear(None, n_out),
#         )
#         self.train = True
#
#     def __call__(self, x):
#         h = F.max_pooling_2d(F.local_response_normalization(
#             F.relu(self.conv1(x))), 3, stride=2)
#         h = F.max_pooling_2d(F.local_response_normalization(
#             F.relu(self.conv2(h))), 3, stride=2)
#         h = F.relu(self.conv3(h))
#         h = F.relu(self.conv4(h))
#         h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
#         h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
#         h = F.dropout(F.relu(self.my_fc7(h)), train=self.train)
#         h = self.my_fc8(h)
#         return h
