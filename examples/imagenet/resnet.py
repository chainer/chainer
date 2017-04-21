#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

from cupy.prof import time_range


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, nobias=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, nobias=True),

            # bn4=L.BatchNormalization(in_size),
            conv4=L.Convolution2D(in_size, out_size, 1,
                                  stride, 0, nobias=True),
        )

    def __call__(self, x):
        h = F.relu(self.bn1(x))
        h1 = self.conv1(h)
        h1 = self.conv2(F.relu(self.bn2(h1)))
        h1 = self.conv3(F.relu(self.bn3(h1)))

        h2 = self.conv4(h)

        return h1 + h2


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, nobias=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, nobias=True),
        )

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        h = self.conv3(F.relu(self.bn3(h)))

        return h + x


class Block(chainer.Chain):

    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        links = [('a', BottleNeckA(in_size, ch, out_size, stride))]
        for i in range(layer - 1):
            links += [('b{}'.format(i + 1), BottleNeckB(out_size, ch))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x):
        for name, _ in sorted(self.forward):
            f = getattr(self, name)
            x = f(x)

        return x


class ResNet(chainer.Chain):

    insize = 224

    def __init__(self, r2, r3, r4, r5):
        super(ResNet, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block(r2, 64, 64, 256, 1),
            res3=Block(r3, 256, 128, 512),
            res4=Block(r4, 512, 256, 1024),
            res5=Block(r5, 1024, 512, 2048),
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
        pred = F.softmax(h)
        accuracy = F.accuracy(pred, t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


class ResNet_prof(ResNet):

    def __call__(self, x, t):
        with time_range("FWD"):
            h = self.bn1(self.conv1(x))
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)
            with time_range("RES2", color_id=1):
                h = self.res2(h)
            with time_range("RES3", color_id=2):
                h = self.res3(h)
            with time_range("RES4", color_id=3):
                h = self.res4(h)
            with time_range("RES5", color_id=4):
                h = self.res5(h)
            h = F.average_pooling_2d(h, 7, stride=1)
            h = self.fc(h)

        loss = F.softmax_cross_entropy(h, t)
        pred = F.softmax(h)
        accuracy = F.accuracy(pred, t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


class ResNet50(ResNet):

    def __init__(self):
        super(ResNet50, self).__init__(3, 4, 6, 3)


class ResNet50_prof(ResNet_prof):

    def __init__(self):
        super(ResNet50_prof, self).__init__(3, 4, 6, 3)


class ResNet152(ResNet):

    def __init__(self):
        super(ResNet152, self).__init__(3, 8, 36, 3)


class ResNet152_prof(ResNet_prof):

    def __init__(self):
        super(ResNet152_prof, self).__init__(3, 8, 36, 3)
