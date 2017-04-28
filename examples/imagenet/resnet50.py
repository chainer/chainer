# Original author: yasunorikudo
# (https://github.com/yasunorikudo/chainer-ResNet)

import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0,
                                  initialW=initializers.HeNormal()
                                  nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1,
                                  initialW=initializers.HeNormal(),
                                  nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0,
                                  initialW=initializers.HeNormal(),
                                  nobias=True),
            bn3=L.BatchNormalization(out_size),

            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0,
                                  initialW=initializers.HeNormal(),
                                  nobias=True),
            bn4=L.BatchNormalization(out_size),
        )

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))

        return F.relu(h1 + h2)


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__(
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0,
                                  initialW=initializers.HeNormal(),
                                  nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1,
                                  initialW=initializers.HeNormal(),
                                  nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0,
                                  initialW=initializers.HeNormal(),
                                  nobias=True),
            bn3=L.BatchNormalization(in_size),
        )

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)


class Block(chainer.ChainList):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.add_link(BottleNeckA(in_size, ch, out_size, stride))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class ResNet50(chainer.Chain):

    insize = 224

    def __init__(self):
        super(ResNet50, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3,
                                  initialW=initializers.HeNormal())
            bn1=L.BatchNormalization(64),
            res2=Block(3, 64, 64, 256, 1),
            res3=Block(4, 256, 128, 512),
            res4=Block(6, 512, 256, 1024),
            res5=Block(3, 1024, 512, 2048),
            fc=L.Linear(2048, 1000),
        )

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x))
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
