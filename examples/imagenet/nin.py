import math

import chainer
import chainer.functions as F
import chainer.links as L


class NIN(chainer.DictLink):

    """Network-in-Network example model."""

    insize = 227

    def __init__(self):
        w = math.sqrt(2)  # MSRA scaling
        super(NIN, self).__init__(
            l1=L.MLPConvolution2D(3, [96, 96, 96], 11, wscale=w, stride=4),
            l2=L.MLPConvolution2D(96, [256, 256, 256], 5, wscale=w, pad=2),
            l3=L.MLPConvolution2D(256, [384, 384, 384], 3, wscale=w),
            l4=L.MLPConvolution2D(384, [1024, 1024, 1000], 3, wscale=w),
        )

    def forward(self, x, t, train=True):
        h = F.relu(self['l1'](x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self['l2'](x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.relu(self['l3'](x))
        h = F.max_pooling_2d(h, 3, stride=2)
        h = F.dropout(h, train=train)
        h = F.relu(self['l4'](x))
        h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 1000))
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
