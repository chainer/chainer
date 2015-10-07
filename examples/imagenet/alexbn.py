import chainer
import chainer.functions as F
import chainer.links as L


class AlexBN(chainer.DictLink):

    """Single-GPU AlexNet with Normalization layers replaced by BatchNormalization.

    """

    insize = 227

    def __init__(self):
        super(AlexBN, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4),
            bn1=L.BatchNormalization(96),
            conv2=L.Convolution2D(96, 256,  5, pad=2),
            bn2=L.BatchNormalization(256),
            conv3=L.Convolution2D(256, 384,  3, pad=1),
            conv4=L.Convolution2D(384, 384,  3, pad=1),
            conv5=L.Convolution2D(384, 256,  3, pad=1),
            fc6=L.Linear(9216, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )

    def forward(self, x, t, train=True):
        h = F.max_pooling_2d(
            F.relu(self['bn1'](self['conv1'](x))), 3, stride=2)
        h = F.max_pooling_2d(
            F.relu(self['bn2'](self['conv2'](h))), 3, stride=2)
        h = F.relu(self['conv3'](h))
        h = F.relu(self['conv4'](h))
        h = F.max_pooling_2d(F.relu(self['conv5'](h)), 3, stride=2)
        h = F.dropout(F.relu(self['fc6'](h)))
        h = F.dropout(F.relu(self['fc7'](h)))
        h = self['fc8'](h)
        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
