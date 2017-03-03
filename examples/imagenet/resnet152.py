#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import math


class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),

            bn4=L.BatchNormalization(in_size),
            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w,
                                  nobias=True),
        )

    def __call__(self, x, train):
        h1 = self.conv1(F.relu(self.bn1(x, test=not train)))
        h1 = self.conv2(F.relu(self.bn2(h1, test=not train)))
        h1 = self.conv3(F.relu(self.bn3(h1, test=not train)))

        h2 = self.conv4(F.relu(self.bn4(x, test=not train)))

        return h1 + h2


class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True),
        )

    def __call__(self, x, train):
        h = self.conv1(F.relu(self.bn1(x, test=not train)))
        h = self.conv2(F.relu(self.bn2(h, test=not train)))
        h = self.conv3(F.relu(self.bn3(h, test=not train)))

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

    def __call__(self, x, train):
        for name, _ in sorted(self.forward):
            f = getattr(self, name)
            x = f(x, train)

        return x


class ResNet(chainer.Chain):

    insize = 224

    def __init__(self):
        w = math.sqrt(2)
        super(ResNet, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block(3, 64, 64, 256, 1),
            res3=Block(8, 256, 128, 512),
            res4=Block(36, 512, 256, 1024),
            res5=Block(3, 1024, 512, 2048),
            fc=L.Linear(2048, 1000),
        )
        self.train = True

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        loss = F.softmax_cross_entropy(h, t)
        pred = F.softmax(h)
        accuracy = F.accuracy(pred, t)
        chainer.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


class ResNet_recomp(ResNet):

    insize = 224

    def __call__(self, x, t):
        with chainer.use_recompute("RELU", "BN"):
            return ResNet.__call__(self, x, t)
