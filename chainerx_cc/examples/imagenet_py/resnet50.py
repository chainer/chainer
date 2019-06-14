#!/usr/bin/env python3

import numpy as np

import chainerx as chx


class Convolution2D(object):

    def __init__(self, in_channels, out_channels, ksize, stride, pad,
                 initialW=None, nobias=False, groups=1):
        W_shape = out_channels, int(in_channels / groups), ksize, ksize
        self.W = chx.array(np.random.normal(size=W_shape).astype(np.float32))
        if nobias:
            self.b = None
        else:
            self.b = chx.array(np.random.normal(
                size=out_channels).astype(np.float32))
        self.stride = stride
        self.pad = pad

    @property
    def params(self):
        return self.W, self.b

    def __call__(self, x):
        if self.b is not None:
            return chx.conv(
                x, self.W, self.b, stride=self.stride, pad=self.pad)
        else:
            return chx.conv(x, self.W, stride=self.stride, pad=self.pad)

    def require_grad(self):
        for param in self.params:
            if param is not None:
                param.require_grad()

    def update(self, lr):
        for param in self.params:
            if param is not None:
                p = param.as_grad_stopped()
                p -= lr * param.grad.as_grad_stopped()
                param.cleargrad()


class BatchNormalization(object):

    def __init__(self, size, dtype=chx.float32):
        shape = size,
        self.avg_mean = chx.zeros(shape, dtype)
        self.avg_var = chx.zeros(shape, dtype)
        self.gamma = chx.ones(shape, dtype)
        self.beta = chx.zeros(shape, dtype)

    def __call__(self, x):
        return chx.batch_norm(x, self.gamma, self.beta,
                              running_mean=self.avg_mean,
                              running_var=self.avg_var,
                              axis=(0, 2, 3))

    @property
    def params(self):
        return self.gamma, self.beta

    def require_grad(self):
        for param in self.params:
            param.require_grad()

    def update(self, lr):
        for param in self.params:
            p = param.as_grad_stopped()
            p -= lr * param.grad.as_grad_stopped()
            param.cleargrad()


class Linear(object):

    def __init__(self, n_in, n_out):
        W = np.random.randn(n_in, n_out).astype(np.float32)
        W /= np.sqrt(n_in)
        self.W = chx.array(W)
        self.b = chx.zeros((n_out,), dtype=chx.float32)

    def __call__(self, x):
        x = x.reshape(x.shape[:2])
        return x.dot(self.W) + self.b

    @property
    def params(self):
        return self.W, self.b

    def require_grad(self):
        for param in self.params:
            param.require_grad()

    def update(self, lr):
        for param in self.params:
            p = param.as_grad_stopped()
            p -= lr * param.grad.as_grad_stopped()
            param.cleargrad()


class BottleNeckA(object):

    def __init__(self, in_size, ch, out_size, stride=2, groups=1):

        initialW = None

        self.conv1 = Convolution2D(
            in_size, ch, 1, stride, 0, initialW=initialW, nobias=True)
        self.bn1 = BatchNormalization(ch)
        self.conv2 = Convolution2D(
            ch, ch, 3, 1, 1, initialW=initialW, nobias=True, groups=groups)
        self.bn2 = BatchNormalization(ch)
        self.conv3 = Convolution2D(
            ch, out_size, 1, 1, 0, initialW=initialW, nobias=True)
        self.bn3 = BatchNormalization(out_size)
        self.conv4 = Convolution2D(
            in_size, out_size, 1, stride, 0, initialW=initialW, nobias=True)
        self.bn4 = BatchNormalization(out_size)

    def __call__(self, x):
        h1 = chx.maximum(0, self.bn1(self.conv1(x)))
        h1 = chx.maximum(0, self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return chx.maximum(0, h1 + h2)

    @property
    def params(self):
        return (self.conv1, self.bn1, self.conv2, self.bn2,
                self.conv3, self.bn3, self.conv4, self.bn4)

    def require_grad(self):
        for param in self.params:
            param.require_grad()

    def update(self, lr):
        for param in self.params:
            param.update(lr)


class BottleNeckB(object):

    def __init__(self, in_size, ch, groups=1):
        initialW = None

        self.conv1 = Convolution2D(
            in_size, ch, 1, 1, 0, initialW=initialW, nobias=True)
        self.bn1 = BatchNormalization(ch)
        self.conv2 = Convolution2D(
            ch, ch, 3, 1, 1, initialW=initialW, nobias=True, groups=groups)
        self.bn2 = BatchNormalization(ch)
        self.conv3 = Convolution2D(
            ch, in_size, 1, 1, 0, initialW=initialW, nobias=True)
        self.bn3 = BatchNormalization(in_size)

    def __call__(self, x):
        h = chx.maximum(0, self.bn1(self.conv1(x)))
        h = chx.maximum(0, self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return chx.maximum(0, h + x)

    @property
    def params(self):
        return self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3

    def require_grad(self):
        for param in self.params:
            param.require_grad()

    def update(self, lr):
        for param in self.params:
            param.update(lr)


class Block(object):

    def __init__(self, layer, in_size, ch, out_size, stride=2, groups=1):
        self.children = []
        self.add_link(BottleNeckA(in_size, ch, out_size, stride, groups))
        for i in range(layer - 1):
            self.add_link(BottleNeckB(out_size, ch, groups))

    def __call__(self, x):
        for f in self.children:
            x = f(x)
        return x

    def add_link(self, x):
        self.children.append(x)

    def require_grad(self):
        for child in self.children:
            child.require_grad()

    def update(self, lr):
        for child in self.children:
            child.update(lr)


class ResNet50(object):

    insize = 224

    def __init__(self):
        self.conv1 = Convolution2D(3, 64, 7, 2, 3)
        self.bn1 = BatchNormalization(64)
        self.res2 = Block(3, 64, 64, 256, 1)
        self.res3 = Block(4, 256, 128, 512)
        self.res4 = Block(6, 512, 256, 1024)
        self.res5 = Block(3, 1024, 512, 2048)
        self.fc = Linear(2048, 1000)

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = chx.max_pool(chx.maximum(0, h), 3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = chx.average_pool(h, 7, stride=1)
        h = self.fc(h)

        return h

    @property
    def params(self):
        return (self.conv1, self.bn1, self.res2, self.res3, self.res4,
                self.res5, self.fc)

    def require_grad(self):
        for param in self.params:
            param.require_grad()

    def update(self, lr):
        for param in self.params:
            param.update(lr)
