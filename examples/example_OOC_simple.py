import numpy

from chainer import Chain
from chainer import optimizers
from chainer import Variable

import chainer.functions as F
import chainer.links as L

import cupy.cuda.nvtx as nvtx
import cupy.cuda.runtime as runtime
import cupy.cuda.stream as stream

############################################################


class MLP_ref(Chain):

    def __init__(self, size1, size2):
        super(MLP_ref, self).__init__(
            l1=L.Linear(size1, size2),
            l2=L.Linear(size2, size1),
            l3=L.Linear(size1, size2),
            l4=L.Linear(size2, size1),
        )

    def __call__(self, x, t):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        y = self.l4(h3)
        loss = F.softmax_cross_entropy(y, t)
        return loss

############################################################


class MLP_ooc(Chain):

    def __init__(self, size1, size2):
        super(MLP_ooc, self).__init__(
            l1=L.Linear(size1, size2, forget_x=True),
            l2=L.Linear(size2, size1, forget_x=True),
            l3=L.Linear(size1, size2, forget_x=True),
            l4=L.Linear(size2, size1, forget_x=True),
        )
        self.disable_swapout_params()

    def __call__(self, x, t):
        runtime.deviceSynchronize()
        nvtx.RangePush("Forward")

        h1 = self.l1(x)
        h1 = F.relu(h1)
        h2 = self.l2(h1)
        h2.set_end_of_sub_graph()
        h2 = F.relu(h2)
        h3 = self.l3(h2)
        h3 = F.relu(h3)
        y = self.l4(h3)

        runtime.deviceSynchronize()
        nvtx.RangePop()

        loss = F.softmax_cross_entropy(y, t)
        return loss

############################################################

size1 = 1001
size2 = 999

# size1 = 11
# size2 = 9

############################################################

model1 = MLP_ref(size1, size2)
opt1 = optimizers.SGD()
opt1.setup(model1)

############################################################

model2 = MLP_ooc(size1, size2)
model2.disable_swapout_params()
opt2 = optimizers.SGD()
opt2.setup(model2)

############################################################

def compare_link(l1, l2):
    numpy.testing.assert_equal(l1.W.data, l2.W.data)
    numpy.testing.assert_equal(l1.b.data, l2.b.data)


def compare_links():
    compare_link(model1.l1, model2.l1)
    compare_link(model1.l2, model2.l2)
    compare_link(model1.l3, model2.l3)
    compare_link(model1.l4, model2.l4)

############################################################

model2.l1.W.copydata(model1.l1.W)
model2.l1.b.copydata(model1.l1.b)

model2.l2.W.copydata(model1.l2.W)
model2.l2.b.copydata(model1.l2.b)

model2.l3.W.copydata(model1.l3.W)
model2.l3.b.copydata(model1.l3.b)

model2.l4.W.copydata(model1.l4.W)
model2.l4.b.copydata(model1.l4.b)

compare_links()

############################################################

nbatch = 1000
# nbatch = 100
# nbatch = 10

x = numpy.random.uniform(-1, 1, (nbatch, size1)).astype(numpy.float32)
x = Variable(x)

x1 = Variable(numpy.zeros_like(x.data))
x2 = Variable(numpy.zeros_like(x.data))
x1.copydata(x)
x2.copydata(x)
x1.name = "x1"
x2.name = "x2"
x1.cleargrad()
x2.cleargrad()

label = numpy.zeros((nbatch), dtype=numpy.int32)
label = Variable(label)
# print(label.data)

############################################################

# num_loop = 10
num_loop = 5

print("#################### reference model ####################")
if True:
    model1.to_gpu()

    x1.to_gpu()
    label.to_gpu()

    for loop in range(0, num_loop):
        loss1 = model1(x1, label)
        print("loop:{}, loss:{}".format(loop, loss1.data))

        model1.cleargrads()
        loss1.backward()
        opt1.update()

    model1.to_cpu()


print("#################### OOC model ####################")
if True:
    model2.to_gpu()

    x2.to_gpu()
    label.to_gpu()

    for loop in range(0, num_loop):
        loss2 = model2(x2, label)
        print("loop:{}, loss:{}".format(loop, loss2.data))

        model2.cleargrads()
        loss2.backward()
        opt2.update()

    model2.to_cpu()

print("########## check variables ##########")

compare_links()
