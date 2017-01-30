import sys
import chainer
from chainer import cuda, Function, gradient_check, utils, Variable
from chainer import optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import math

import numpy

import chainer.functions.util.fused_function as FF

use_GPU = True

if (use_GPU):
    import cupy as xp
else:
    import numpy as xp

import cupy.cuda.nvtx as nvtx
import cupy.cuda.stream as stream
import cupy.cuda.runtime as runtime

import time
from time import sleep

# set workspace size for cuDNN
_free_mem, total_mem = cuda.cupy.cuda.runtime.memGetInfo()
# size = long(total_mem * 0.1)
# size = long(total_mem * 0.01)
size = long(total_mem * 0.05)
cuda.set_max_workspace_size(size)

############################################################

class BottleNeckA_ref(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA_ref, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True),

            bn4=L.BatchNormalization(in_size),
            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True),
        )

    def __call__(self, x, train):
        h1 = self.conv1(F.relu(self.bn1(x, test=not train)))
        h1 = self.conv2(F.relu(self.bn2(h1, test=not train)))
        h1 = self.conv3(F.relu(self.bn3(h1, test=not train)))

        h2 = self.conv4(F.relu(self.bn4(x, test=not train)))

        return h1 + h2


class BottleNeckB_ref(chainer.Chain):
    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB_ref, self).__init__(
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


class Block_ref(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block_ref, self).__init__()
        links = [('a', BottleNeckA_ref(in_size, ch, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB_ref(out_size, ch))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, train)

        return x


class ResNet152_ref(chainer.Chain):

    insize = 224

    def __init__(self):
        w = math.sqrt(2)
        super(ResNet152_ref, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block_ref(3, 64, 64, 256, 1),
            res3=Block_ref(8, 256, 128, 512),
            res4=Block_ref(36, 512, 256, 1024),
            res5=Block_ref(3, 1024, 512, 2048),
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

        if self.train:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss
        else:
            return h

############################################################


class BottleNeckA_rmem(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA_rmem, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True, forget_x=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True, forget_x=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True, forget_x=True),

            bn4=L.BatchNormalization(in_size),
            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True, forget_x=True),
        )

    def __call__(self, x, train):
        h1 = self.conv1(F.relu(self.bn1(x , test=not train), forget_x=True))
        h1 = self.conv2(F.relu(self.bn2(h1, test=not train), forget_x=True))
        h1 = self.conv3(F.relu(self.bn3(h1, test=not train), forget_x=True))

        h2 = self.conv4(F.relu(self.bn4(x, test=not train)))

        return h1 + h2

class BottleNeckB_rmem(chainer.Chain):
    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB_rmem, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True, forget_x=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True, forget_x=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True, forget_x=True),
        )

    def __call__(self, x, train):
        h = self.conv1(F.relu(self.bn1(x, test=not train), forget_x=True))
        h = self.conv2(F.relu(self.bn2(h, test=not train), forget_x=True))
        h = self.conv3(F.relu(self.bn3(h, test=not train), forget_x=True))

        return h + x


class Block_rmem(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block_rmem, self).__init__()
        links = [('a', BottleNeckA_rmem(in_size, ch, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB_rmem(out_size, ch))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, train)

        return x


class ResNet152_rmem(chainer.Chain):

    insize = 224

    def __init__(self):
        w = math.sqrt(2)
        super(ResNet152_rmem, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64),
            res2=Block_rmem(3, 64, 64, 256, 1),
            res3=Block_rmem(8, 256, 128, 512),
            res4=Block_rmem(36, 512, 256, 1024),
            res5=Block_rmem(3, 1024, 512, 2048),
            fc=L.Linear(2048, 1000),
        )
        self.train = True

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h), 3, stride=2, forget_x=True)
        h = self.res2(h, self.train)
        h = self.res3(h, self.train)
        h = self.res4(h, self.train)
        h = self.res5(h, self.train)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        if self.train:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss
        else:
            return h

############################################################

class BottleNeckA_ooc(chainer.Chain):
    def __init__(self, in_size, ch, out_size, stride=2):
        w = math.sqrt(2)
        super(BottleNeckA_ooc, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, stride, 0, w, nobias=True, forget_x=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True, forget_x=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_size, 1, 1, 0, w, nobias=True, forget_x=True),

            bn4=L.BatchNormalization(in_size),
            conv4=L.Convolution2D(in_size, out_size, 1, stride, 0, w, nobias=True, forget_x=True),
        )

    def __call__(self, x, train):
        h1 = self.conv1(F.relu(self.bn1(x , test=not train), forget_x=True))
        h1 = self.conv2(F.relu(self.bn2(h1, test=not train), forget_x=True))
        h1 = self.conv3(F.relu(self.bn3(h1, test=not train), forget_x=True))

        h2 = self.conv4(F.relu(self.bn4(x, test=not train), forget_x=True))

        y = h1 + h2
        h1.forget()
        h2.forget()

        return y

class BottleNeckB_ooc(chainer.Chain):
    def __init__(self, in_size, ch):
        w = math.sqrt(2)
        super(BottleNeckB_ooc, self).__init__(
            bn1=L.BatchNormalization(in_size),
            conv1=L.Convolution2D(in_size, ch, 1, 1, 0, w, nobias=True, forget_x=True),
            bn2=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, 1, 1, w, nobias=True, forget_x=True),
            bn3=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, in_size, 1, 1, 0, w, nobias=True, forget_x=True),
        )

    def __call__(self, x, train):
        h = self.conv1(F.relu(self.bn1(x, test=not train), forget_x=True))
        h = self.conv2(F.relu(self.bn2(h, test=not train), forget_x=True))
        h = self.conv3(F.relu(self.bn3(h, test=not train), forget_x=True))

        y = h + x
        h.forget()

        return y


class Block_ooc(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(Block_ooc, self).__init__()
        links = [('a', BottleNeckA_ooc(in_size, ch, out_size, stride))]
        for i in range(layer-1):
            links += [('b{}'.format(i+1), BottleNeckB_ooc(out_size, ch))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, train, stream=None):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, train)
            x.set_end_of_sub_graph(stream=stream)

        return x


class ResNet152_ooc(chainer.Chain):

    insize = 224

    def __init__(self):
        w = math.sqrt(2)
        super(ResNet152_ooc, self).__init__(
            conv1=L.Convolution2D(3, 64, 7, 2, 3, w, nobias=True),
            bn1=L.BatchNormalization(64, forget_x=True),
            res2=Block_ooc(3, 64, 64, 256, 1),
            res3=Block_ooc(8, 256, 128, 512),
            res4=Block_ooc(36, 512, 256, 1024),
            res5=Block_ooc(3, 1024, 512, 2048),
            fc=L.Linear(2048, 1000),
        )
        self.train = True
        self.disable_swapout_params()
        self.stream = None

    def __call__(self, x, t):
        h = self.bn1(self.conv1(x), test=not self.train)
        h = F.max_pooling_2d(F.relu(h, forget_x=True), 3, stride=2, forget_x=True)
        h.set_end_of_sub_graph(stream=self.stream)
        h = self.res2(h, self.train, self.stream)
        h = self.res3(h, self.train, self.stream)
        h = self.res4(h, self.train, self.stream)
        h = self.res5(h, self.train, self.stream)
        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.fc(h)

        if self.stream is None:
            self.stream = stream.Stream(non_blocking=True)

        if self.train:
            loss = F.softmax_cross_entropy(h, t)
            chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
            return loss
        else:
            return h

############################################################

model = ResNet152_ref()
# model = ResNet152_rmem()
# model = ResNet152_ooc()

opt = optimizers.SGD()
opt.setup(model)

############################################################

nbatch=10
# nbatch=20
# nbatch=40
# nbatch=60
# nbatch=80
# nbatch=100
# nbatch=120
# nbatch=140
# nbatch=160
# nbatch=180
# nbatch=200

############################################################

x = xp.random.uniform(-1, 1, (nbatch, 3, 224, 224)).astype(xp.float32)
x = Variable( xp.asarray(x) )

x0 = Variable( xp.zeros_like( x.data ) )
x0.copydata(x)
x0.name = "x0"
x0.cleargrad()

label = xp.zeros((nbatch), dtype=xp.int32)
for i in range(0, len(label)):
    label[i] = i % 1000
label = Variable( xp.asarray(label) )

x0.to_gpu()
label.to_gpu()

############################################################

num_loop = 10

############################################################

if True:
    if (use_GPU):
        model.to_gpu()
    
    sleep(1)  #

    accum_t_f = 0
    accum_t_b = 0

    for loop in range(0, num_loop):
    
        runtime.deviceSynchronize()
        nvtx.RangePush("Run: {}".format(loop), loop)
        t0 = time.clock()
    
        model.cleargrads()

        nvtx.RangePush("Forward",1)
        loss = model(x0, label)
        runtime.deviceSynchronize()
        nvtx.RangePop()
        t1 = time.clock()
    
        print 'loop:{}, loss:{}'.format(loop, loss.data)
    
        nvtx.RangePush("Backward & Update",2)
        loss.backward()
        opt.update()
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        runtime.deviceSynchronize()
        nvtx.RangePop()
        t2 = time.clock()

        t_f = t1 - t0
        t_b = t2 - t1

        # print("time: total:{}, forward:{}, backward:{}".format(t_f+t_b, t_f, t_b))

        if loop >= 2:
            accum_t_f += t_f
            accum_t_b += t_b

        sleep(1)  #

    ave_t_f = accum_t_f / (num_loop - 2)
    ave_t_b = accum_t_b / (num_loop - 2)

    print("nbatch:{}".format(nbatch))
    print("average time: total:{}, forward:{}, backward:{}".format(ave_t_f+ave_t_b, ave_t_f, ave_t_b))
    
    if (use_GPU):
        model.to_cpu()

