#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L

# U-net https://arxiv.org/pdf/1611.07004v1.pdf

# convolution-batchnormalization-(dropout)-relu


class ConvBNR(chainer.Chain):
    def __init__(self, ch0, ch1, use_bn=True,
                 sample='down', activation=F.relu, dropout=False):
        self.use_bn = use_bn
        self.activation = activation
        self.dropout = dropout
        w = chainer.initializers.Normal(0.02)
        super(ConvBNR, self).__init__()
        with self.init_scope():
            if sample == 'down':
                self.c = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            else:
                self.c = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
            if use_bn:
                self.bn = L.BatchNormalization(ch1)

    def forward(self, x):
        h = self.c(x)
        if self.use_bn:
            h = self.bn(h)
        if self.dropout:
            h = F.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class Encoder(chainer.Chain):
    def __init__(self, in_ch):
        w = chainer.initializers.Normal(0.02)
        super(Encoder, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
            self.c1 = ConvBNR(64, 128, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)
            self.c2 = ConvBNR(128, 256, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)
            self.c3 = ConvBNR(256, 512, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)
            self.c4 = ConvBNR(512, 512, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)
            self.c5 = ConvBNR(512, 512, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)
            self.c6 = ConvBNR(512, 512, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)
            self.c7 = ConvBNR(512, 512, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)

    def forward(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1, 8):
            hs.append(self['c%d' % i](hs[i-1]))
        return hs


class Decoder(chainer.Chain):
    def __init__(self, out_ch):
        w = chainer.initializers.Normal(0.02)
        super(Decoder, self).__init__()
        with self.init_scope():
            self.c0 = ConvBNR(512, 512, use_bn=True, sample='up',
                              activation=F.relu, dropout=True)
            self.c1 = ConvBNR(1024, 512, use_bn=True,
                              sample='up', activation=F.relu, dropout=True)
            self.c2 = ConvBNR(1024, 512, use_bn=True,
                              sample='up', activation=F.relu, dropout=True)
            self.c3 = ConvBNR(1024, 512, use_bn=True,
                              sample='up', activation=F.relu, dropout=False)
            self.c4 = ConvBNR(1024, 256, use_bn=True,
                              sample='up', activation=F.relu, dropout=False)
            self.c5 = ConvBNR(512, 128, use_bn=True, sample='up',
                              activation=F.relu, dropout=False)
            self.c6 = ConvBNR(256, 64, use_bn=True, sample='up',
                              activation=F.relu, dropout=False)
            self.c7 = L.Convolution2D(128, out_ch, 3, 1, 1, initialW=w)

    def forward(self, hs):
        h = self.c0(hs[-1])
        for i in range(1, 8):
            h = F.concat([h, hs[-i-1]])
            if i < 7:
                h = self['c%d' % i](h)
            else:
                h = self.c7(h)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        w = chainer.initializers.Normal(0.02)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = ConvBNR(in_ch, 32, use_bn=False, sample='down',
                                activation=F.leaky_relu, dropout=False)
            self.c0_1 = ConvBNR(out_ch, 32, use_bn=False, sample='down',
                                activation=F.leaky_relu, dropout=False)
            self.c1 = ConvBNR(64, 128, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)
            self.c2 = ConvBNR(128, 256, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)
            self.c3 = ConvBNR(256, 512, use_bn=True, sample='down',
                              activation=F.leaky_relu, dropout=False)
            self.c4 = L.Convolution2D(512, 1, 3, 1, 1, initialW=w)

    def forward(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        return h
