import sys
import chainer
from chainer import cuda, Function, gradient_check, utils, Variable
from chainer import optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import math

import numpy

use_GPU = True
#use_GPU = False

if (use_GPU):
    import cupy as xp
else:
    import numpy as xp

import cupy.cuda.nvtx as nvtx
import cupy.cuda.stream as stream
import cupy.cuda.runtime as runtime

from time import sleep

# set workspace size for cuDNN
_free_mem, total_mem = cuda.cupy.cuda.runtime.memGetInfo()
# size = long(total_mem * 0.1)
size = long(total_mem * 0.01)
cuda.set_max_workspace_size(size)

############################################################

class VGG16_ref(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_ref, self).__init__(
            conv1_1=L.Convolution2D(  3,  64, 3, pad=1, use_cudnn=use_cudnn),
            conv1_2=L.Convolution2D( 64,  64, 3, pad=1, use_cudnn=use_cudnn),
            conv2_1=L.Convolution2D( 64, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv3_1=L.Convolution2D(128, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv4_1=L.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_1=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, x, t):
        h1_1 = F.relu(self.conv1_1(x   ), use_cudnn=self.use_cudnn)
        h1_2 = F.relu(self.conv1_2(h1_1), use_cudnn=self.use_cudnn)
        h1   = F.max_pooling_2d(h1_2, 2, stride=2, use_cudnn=self.use_cudnn)
        h2_1 = F.relu(self.conv2_1(h1  ), use_cudnn=self.use_cudnn)
        h2_2 = F.relu(self.conv2_2(h2_1), use_cudnn=self.use_cudnn)
        h2   = F.max_pooling_2d(h2_2, 2, stride=2, use_cudnn=self.use_cudnn)
        h3_1 = F.relu(self.conv3_1(h2  ), use_cudnn=self.use_cudnn)
        h3_2 = F.relu(self.conv3_2(h3_1), use_cudnn=self.use_cudnn)
        h3_3 = F.relu(self.conv3_3(h3_2), use_cudnn=self.use_cudnn)
        h3   = F.max_pooling_2d(h3_3, 2, stride=2, use_cudnn=self.use_cudnn)
        h4_1 = F.relu(self.conv4_1(h3  ), use_cudnn=self.use_cudnn)
        h4_2 = F.relu(self.conv4_2(h4_1), use_cudnn=self.use_cudnn)
        h4_3 = F.relu(self.conv4_3(h4_2), use_cudnn=self.use_cudnn)
        h4   = F.max_pooling_2d(h4_3, 2, stride=2, use_cudnn=self.use_cudnn)
        h5_1 = F.relu(self.conv5_1(h4  ), use_cudnn=self.use_cudnn)
        h5_2 = F.relu(self.conv5_2(h5_1), use_cudnn=self.use_cudnn)
        h5_3 = F.relu(self.conv5_3(h5_2), use_cudnn=self.use_cudnn)
        h5   = F.max_pooling_2d(h5_3, 2, stride=2, use_cudnn=self.use_cudnn)
        h6   = F.relu(self.fc6(h5), use_cudnn=self.use_cudnn)
        h7   = F.relu(self.fc7(h6), use_cudnn=self.use_cudnn)
        h8   = self.fc8(h7)
        loss = F.softmax_cross_entropy(h8, t)
        return loss

############################################################

class VGG16_rmem(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_rmem, self).__init__(
            conv1_1=L.Convolution2D(  3,  64, 3, pad=1, use_cudnn=use_cudnn),
            conv1_2=L.Convolution2D( 64,  64, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv2_1=L.Convolution2D( 64, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv3_1=L.Convolution2D(128, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv4_1=L.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv4_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv5_1=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv5_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )
        self.use_cudnn = use_cudnn

    def __call__(self, x, t):
        h1_1 = self.conv1_1(x   )
        h1_2 = self.conv1_2(h1_1)
        h1   = F.max_pooling_2d(h1_2, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h2_1 = self.conv2_1(h1  )
        h2_2 = self.conv2_2(h2_1)
        h2   = F.max_pooling_2d(h2_2, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h3_1 = self.conv3_1(h2  )
        h3_2 = self.conv3_2(h3_1)
        h3_3 = self.conv3_3(h3_2)
        h3   = F.max_pooling_2d(h3_3, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h4_1 = self.conv4_1(h3  )
        h4_2 = self.conv4_2(h4_1)
        h4_3 = self.conv4_3(h4_2)
        h4   = F.max_pooling_2d(h4_3, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h5_1 = self.conv5_1(h4  )
        h5_2 = self.conv5_2(h5_1)
        h5_3 = self.conv5_3(h5_2)
        h5   = F.max_pooling_2d(h5_3, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h6   = F.relu(self.fc6(h5), use_cudnn=self.use_cudnn)
        h7   = F.relu(self.fc7(h6), use_cudnn=self.use_cudnn)
        h8   = self.fc8(h7)
        loss = F.softmax_cross_entropy(h8, t)
        return loss

############################################################

class VGG16_ooc(chainer.Chain):
    def __init__(self, use_cudnn):
        super(VGG16_ooc, self).__init__(
            conv1_1=L.Convolution2D(  3,  64, 3, pad=1, use_cudnn=use_cudnn),
            conv1_2=L.Convolution2D( 64,  64, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv2_1=L.Convolution2D( 64, 128, 3, pad=1, use_cudnn=use_cudnn),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv3_1=L.Convolution2D(128, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv4_1=L.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv4_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv5_1=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv5_2=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            conv5_3=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn, pre_func=F.ReLU()),
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000),
        )
        self.use_cudnn = use_cudnn
        # self.stream = stream.Stream(non_blocking=True)
        self.stream = None
        self.disable_swapout_params()

    def __call__(self, x, t):
        h1_1 = self.conv1_1(x   )
        h1_1.set_end_of_sub_graph(stream=self.stream)
        h1_2 = self.conv1_2(h1_1)
        h1_2.set_end_of_sub_graph(stream=self.stream)
        h1   = F.max_pooling_2d(h1_2, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h1.set_end_of_sub_graph(stream=self.stream)
        h2_1 = self.conv2_1(h1  )
        h2_1.set_end_of_sub_graph(stream=self.stream)
        h2_2 = self.conv2_2(h2_1)
        h2_2.set_end_of_sub_graph(stream=self.stream)
        h2   = F.max_pooling_2d(h2_2, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h2.set_end_of_sub_graph(stream=self.stream)
        h3_1 = self.conv3_1(h2  )
        h3_1.set_end_of_sub_graph(stream=self.stream)
        h3_2 = self.conv3_2(h3_1)
        h3_2.set_end_of_sub_graph(stream=self.stream)
        h3_3 = self.conv3_3(h3_2)
        h3_3.set_end_of_sub_graph(stream=self.stream)
        h3   = F.max_pooling_2d(h3_3, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h3.set_end_of_sub_graph(stream=self.stream)
        h4_1 = self.conv4_1(h3  )
        h4_1.set_end_of_sub_graph(stream=self.stream)
        h4_2 = self.conv4_2(h4_1)
        h4_2.set_end_of_sub_graph(stream=self.stream)
        h4_3 = self.conv4_3(h4_2)
        h4_3.set_end_of_sub_graph(stream=self.stream)
        h4   = F.max_pooling_2d(h4_3, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h4.set_end_of_sub_graph(stream=self.stream)
        h5_1 = self.conv5_1(h4  )
        h5_1.set_end_of_sub_graph(stream=self.stream)
        h5_2 = self.conv5_2(h5_1)
        h5_2.set_end_of_sub_graph(stream=self.stream)
        h5_3 = self.conv5_3(h5_2)
        h5_3.set_end_of_sub_graph(stream=self.stream)
        h5   = F.max_pooling_2d(h5_3, 2, stride=2, use_cudnn=self.use_cudnn, pre_func=F.ReLU())
        h5.set_end_of_sub_graph(stream=self.stream)
        h6   = F.relu(self.fc6(h5), use_cudnn=self.use_cudnn)
        h6.set_end_of_sub_graph(stream=self.stream)
        h7   = F.relu(self.fc7(h6), use_cudnn=self.use_cudnn)
        h7.set_end_of_sub_graph(stream=self.stream)
        h8   = self.fc8(h7)
        loss = F.softmax_cross_entropy(h8, t)

        if self.stream is None:
            self.stream = stream.Stream(non_blocking=True)

        return loss

############################################################

def compare_link(name, l1, l2):
    numpy.testing.assert_equal(l1.W.data, l2.W.data)
    numpy.testing.assert_equal(l1.b.data, l2.b.data)

def compare_links():
    compare_link( 'conv1_1', model1.conv1_1, model2.conv1_1 )

############################################################

model0 = VGG16_ref(use_cudnn=True)
opt0 = optimizers.SGD()
opt0.setup(model0)

############################################################

model1 = VGG16_rmem(use_cudnn=True)
opt1 = optimizers.SGD()
opt1.setup(model1)

############################################################

model2 = VGG16_ooc(use_cudnn=True)
model2.disable_swapout_params()
opt2 = optimizers.SGD()
opt2.setup(model2)

############################################################

model1.conv1_1.W.copydata( model0.conv1_1.W )
model1.conv1_1.b.copydata( model0.conv1_1.b )
model1.conv1_2.W.copydata( model0.conv1_2.W )
model1.conv1_2.b.copydata( model0.conv1_2.b )

model1.conv2_1.W.copydata( model0.conv2_1.W )
model1.conv2_1.b.copydata( model0.conv2_1.b )
model1.conv2_2.W.copydata( model0.conv2_2.W )
model1.conv2_2.b.copydata( model0.conv2_2.b )

model1.conv3_1.W.copydata( model0.conv3_1.W )
model1.conv3_1.b.copydata( model0.conv3_1.b )
model1.conv3_2.W.copydata( model0.conv3_2.W )
model1.conv3_2.b.copydata( model0.conv3_2.b )
model1.conv3_3.W.copydata( model0.conv3_3.W )
model1.conv3_3.b.copydata( model0.conv3_3.b )

model1.conv4_1.W.copydata( model0.conv4_1.W )
model1.conv4_1.b.copydata( model0.conv4_1.b )
model1.conv4_2.W.copydata( model0.conv4_2.W )
model1.conv4_2.b.copydata( model0.conv4_2.b )
model1.conv4_3.W.copydata( model0.conv4_3.W )
model1.conv4_3.b.copydata( model0.conv4_3.b )

model1.conv5_1.W.copydata( model0.conv5_1.W )
model1.conv5_1.b.copydata( model0.conv5_1.b )
model1.conv5_2.W.copydata( model0.conv5_2.W )
model1.conv5_2.b.copydata( model0.conv5_2.b )
model1.conv5_3.W.copydata( model0.conv5_3.W )
model1.conv5_3.b.copydata( model0.conv5_3.b )

model1.fc6.W.copydata( model0.fc6.W )
model1.fc6.b.copydata( model0.fc6.b )
model1.fc7.W.copydata( model0.fc7.W )
model1.fc7.b.copydata( model0.fc7.b )
model1.fc8.W.copydata( model0.fc8.W )
model1.fc8.b.copydata( model0.fc8.b )

############################################################

model2.conv1_1.W.copydata( model1.conv1_1.W )
model2.conv1_1.b.copydata( model1.conv1_1.b )
model2.conv1_2.W.copydata( model1.conv1_2.W )
model2.conv1_2.b.copydata( model1.conv1_2.b )

model2.conv2_1.W.copydata( model1.conv2_1.W )
model2.conv2_1.b.copydata( model1.conv2_1.b )
model2.conv2_2.W.copydata( model1.conv2_2.W )
model2.conv2_2.b.copydata( model1.conv2_2.b )

model2.conv3_1.W.copydata( model1.conv3_1.W )
model2.conv3_1.b.copydata( model1.conv3_1.b )
model2.conv3_2.W.copydata( model1.conv3_2.W )
model2.conv3_2.b.copydata( model1.conv3_2.b )
model2.conv3_3.W.copydata( model1.conv3_3.W )
model2.conv3_3.b.copydata( model1.conv3_3.b )

model2.conv4_1.W.copydata( model1.conv4_1.W )
model2.conv4_1.b.copydata( model1.conv4_1.b )
model2.conv4_2.W.copydata( model1.conv4_2.W )
model2.conv4_2.b.copydata( model1.conv4_2.b )
model2.conv4_3.W.copydata( model1.conv4_3.W )
model2.conv4_3.b.copydata( model1.conv4_3.b )

model2.conv5_1.W.copydata( model1.conv5_1.W )
model2.conv5_1.b.copydata( model1.conv5_1.b )
model2.conv5_2.W.copydata( model1.conv5_2.W )
model2.conv5_2.b.copydata( model1.conv5_2.b )
model2.conv5_3.W.copydata( model1.conv5_3.W )
model2.conv5_3.b.copydata( model1.conv5_3.b )

model2.fc6.W.copydata( model1.fc6.W )
model2.fc6.b.copydata( model1.fc6.b )
model2.fc7.W.copydata( model1.fc7.W )
model2.fc7.b.copydata( model1.fc7.b )
model2.fc8.W.copydata( model1.fc8.W )
model2.fc8.b.copydata( model1.fc8.b )

compare_links()

############################################################

nbatch=16
# nbatch=32
# nbatch=64
# nbatch=128

############################################################

x = xp.random.uniform(-1, 1, (nbatch, 3, 224, 224)).astype(xp.float32)
x = Variable( xp.asarray(x) )

x0 = Variable( xp.zeros_like( x.data ) )
x1 = Variable( xp.zeros_like( x.data ) )
x2 = Variable( xp.zeros_like( x.data ) )
x0.copydata(x)
x1.copydata(x)
x2.copydata(x)
x0.name = "x0"
x1.name = "x1"
x2.name = "x2"
x0.cleargrad()
x1.cleargrad()
x2.cleargrad()

label = xp.zeros((nbatch), dtype=xp.int32)
for i in range(0, len(label)):
    label[i] = i % 1000
label = Variable( xp.asarray(label) )

############################################################

num_loop = 5

############################################################

# if False:
if True:
    print '#################### Reference model ####################'
    nvtx.RangePush("Reference", 0)
    if (use_GPU):
        model0.to_gpu()
    
    sleep(1)

    for loop in range(0, num_loop):
        runtime.deviceSynchronize()
        nvtx.RangePush("Run: {}".format(loop), loop)
    
        model0.cleargrads()
        nvtx.RangePush("Forward",1)
        loss0 = model0(x1, label)
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        print 'loop:{}, loss:{}'.format(loop, loss0.data)
    
        nvtx.RangePush("Backward & Update",2)
        loss0.backward()
        opt0.update()
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        runtime.deviceSynchronize()
        nvtx.RangePop()

        sleep(1)
    
    if (use_GPU):
        model0.to_cpu()
    nvtx.RangePop()

############################################################

# if False:
if True:
    print '#################### Reduced memory usage model ####################'
    nvtx.RangePush("Reduced mem usage", 1)
    if (use_GPU):
        model1.to_gpu()
    
    sleep(1)

    for loop in range(0, num_loop):
        runtime.deviceSynchronize()
        nvtx.RangePush("Run: {}".format(loop), loop)
    
        model1.cleargrads()
        nvtx.RangePush("Forward",1)
        loss1 = model1(x1, label)
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        print 'loop:{}, loss:{}'.format(loop, loss1.data)
    
        nvtx.RangePush("Backward & Update",2)
        loss1.backward()
        opt1.update()
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        runtime.deviceSynchronize()
        nvtx.RangePop()

        sleep(1)
    
    if (use_GPU):
        model1.to_cpu()
    nvtx.RangePop()

############################################################

# if False:
if True:
    print '#################### Out-of-core model ####################'
    nvtx.RangePush("Out-of-core", 2)
    if (use_GPU):
        model2.to_gpu()
    
    sleep(1)

    for loop in range(0, num_loop):
        runtime.deviceSynchronize()
        nvtx.RangePush("Run: {}".format(loop), loop)
    
        model2.cleargrads()
        nvtx.RangePush("Forward",1)
        loss2 = model2(x2, label)
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        print 'loop:{}, loss:{}'.format(loop, loss2.data)
    
        nvtx.RangePush("Backward & Update",2)
        loss2.backward()
        opt2.update()
        runtime.deviceSynchronize()
        nvtx.RangePop()
    
        runtime.deviceSynchronize()
        nvtx.RangePop()

        sleep(1)
    
    if (use_GPU):
        model2.to_cpu()
    nvtx.RangePop()

############################################################
