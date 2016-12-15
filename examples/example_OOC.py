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


class Pipeline(object):
    def __init__(self, class_optimizer):
        self.optimizer = class_optimizer
        self.models = []
        self.opts = []
        self.streams = []
        # self.streams.append(None)
        # self.streams.append(None)
        self.streams.append(stream.Stream(non_blocking=True))
        self.streams.append(stream.Stream(non_blocking=True))
        self.run_count = 0
        self.outputs = []

    def add_model(self, model):
        self.models.append(model)
        opt = self.optimizer()
        opt.setup(model)
        self.opts.append(opt)

    def setup(self):
        num_stage = len(self.models)
        i = 0
        model = self.models[i]
        model.cleargrads()
        model.to_gpu()
        for i in range(1, num_stage):
            model = self.models[i]
            model.cleargrads()
            model.to_swap()

    def finalize(self):
        num_stage = len(self.models)
        for i in range(0, num_stage):
            model = self.models[i]
            model.to_cpu()

    def run(self, x, t):
        nvtx.RangePush("Pipeline: run: {}".format(self.run_count),
                       self.run_count)

        num_stage = len(self.models)
        x.to_gpu()
        t.to_gpu()

        # swap-in data of 1st stage (just in case)
        if True:
            stream = self.streams[0]
            next_model = self.models[0]
            next_model.cleargrads()  # to reduce amount of data transfer
            next_model.to_gpu(stream=stream)

        # Foward
        nvtx.RangePush("Pipeline: Forward", 1)
        next_input = x
        loss = None
        for i in range(0, num_stage):
            # swap-in data of next stage
            stream = self.streams[0]
            if stream is not None:
                stream.synchronize()
            if i < num_stage-1:
                next_model = self.models[i+1]
                next_model.cleargrads()  # to reduce amount of data transfer
                next_model.to_gpu(stream=stream)

            # do forward computation of current stage
            nvtx.RangePush("Pipeline: Forward: Stage: {}".format(i), i)
            cur_model = self.models[i]
            cur_input = next_input
            if i < num_stage-1:
                cur_output = cur_model(cur_input)
            else:
                cur_output = cur_model(cur_input, t)
                loss = cur_output
            runtime.deviceSynchronize()
            cur_output.interrupt_backward()
            nvtx.RangePop()

            # swap-out data of current stage
            stream = self.streams[1]
            if stream is not None:
                stream.synchronize()
            if i < num_stage-1:
                cur_output.ancestors_to_swap(stream=stream)

            self.outputs.append(cur_output)
            next_input = cur_output

        print("loop:{}, loss:{}".format(self.run_count, loss.data))
        next_input = None
        nvtx.RangePop()

        # Backward & Update
        nvtx.RangePush("Pipeline: Backward & Update", 2)
        for i in reversed(range(0, num_stage)):
            # swap-in data of next stage
            stream = self.streams[0]
            if stream is not None:
                stream.synchronize()
            if i > 0:
                next_output = self.outputs[i-1]
                next_output.ancestors_to_gpu(stream=stream)

            # do backward computation of current stage
            nvtx.RangePush("Pipeline: Backward & Update: Stage: {}"
                           .format(i), i)
            cur_output = self.outputs.pop()
            cur_model = self.models[i]
            cur_model.cleargrads()
            cur_output.resume_backward()
            cur_output.backward()
            cur_opt = self.opts[i]
            cur_opt.update()
            runtime.deviceSynchronize()
            cur_output.unchain_backward()
            cur_output = None
            cur_model.cleargrads()
            nvtx.RangePop()

            # swap-out data of current stage
            stream = self.streams[1]
            if stream is not None:
                stream.synchronize()
            if i > 0:
                cur_model.to_swap(stream=stream)

        nvtx.RangePop()

        self.run_count += 1
        nvtx.RangePop()

    def run_sync(self, x, t):
        nvtx.RangePush("Pipeline: run: {}".format(self.run_count),
                       self.run_count)
        num_stage = len(self.models)
        x.to_gpu()

        # Foward
        nvtx.RangePush("Pipeline: Forward", 1)
        next_input = x
        loss = None
        for i in range(0, num_stage):
            nvtx.RangePush("Pipeline: Forward: Stage: {}".format(i), i)

            cur_input = next_input
            cur_model = self.models[i]
            cur_model.to_gpu()

            if i < num_stage-1:
                cur_output = cur_model(cur_input)
            else:
                cur_output = cur_model(cur_input, t)
                loss = cur_output
            cur_output.interrupt_backward()
            cur_output.ancestors_to_swap()
            self.outputs.append(cur_output)

            next_input = cur_output
            nvtx.RangePop()

        print("loop:{}, loss:{}".format(self.run_count, loss.data))
        next_input = None
        nvtx.RangePop()

        # Backward & Update
        nvtx.RangePush("Pipeline: Backward & Update", 2)
        for i in reversed(range(0, num_stage)):
            nvtx.RangePush("Pipeline: Backward & Update: Stage: {}".
                           format(i), i)

            cur_output = self.outputs.pop()
            cur_output.ancestors_to_gpu()
            cur_output.resume_backward()

            cur_model = self.models[i]
            cur_model.cleargrads()
            cur_output.backward()
            cur_opt = self.opts[i]
            cur_opt.update()

            cur_output.unchain_backward()
            cur_output = None

            cur_model.cleargrads()
            cur_model.to_swap()
            nvtx.RangePop()

        nvtx.RangePop()

        self.run_count += 1
        nvtx.RangePop()

############################################################


class MLP(Chain):

    def __init__(self, size1, size2):
        super(MLP, self).__init__(
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


class MLP_A(Chain):

    def __init__(self, size1, size2):
        super(MLP_A, self).__init__(
            l1=L.Linear(size1, size2),
        )

    def __call__(self, x):
        h1_a = self.l1(x)
        h1_a.name = "h1_a.l1"
        h1 = F.relu(h1_a)
        h1.name = "h1.relu"
        return h1


class MLP_B(Chain):

    def __init__(self, size1, size2):
        super(MLP_B, self).__init__(
            l2=L.Linear(size2, size1),
        )

    def __call__(self, h1):
        h2_a = self.l2(h1)
        h2_a.name = "h2_a.l2"
        h2 = F.relu(h2_a)
        h2.name = "h2.relu"
        return h2


class MLP_C(Chain):

    def __init__(self, size1, size2):
        super(MLP_C, self).__init__(
            l3=L.Linear(size1, size2),
        )

    def __call__(self, h2):
        h3_a = self.l3(h2)
        h3_a.name = "h3_a.l3"
        h3 = F.relu(h3_a)
        h3.name = "h3.relu"
        return h3


class MLP_D(Chain):

    def __init__(self, size1, size2):
        super(MLP_D, self).__init__(
            l4=L.Linear(size2, size1),
        )

    def __call__(self, h3, t):
        y = self.l4(h3)
        y.name = "y.l4"
        loss = F.softmax_cross_entropy(y, t)
        loss.name = "loss.sce"
        return loss

############################################################

size1 = 1001
size2 = 999

# size1 = 11
# size2 = 9

############################################################

model1 = MLP(size1, size2)
opt1 = optimizers.SGD()
opt1.setup(model1)

############################################################

model2a = MLP_A(size1, size2)
model2b = MLP_B(size1, size2)
model2c = MLP_C(size1, size2)
model2d = MLP_D(size1, size2)

model2a.disable_swapout_params()
model2b.disable_swapout_params()
model2c.disable_swapout_params()
model2d.disable_swapout_params()

pipeline = Pipeline(optimizers.SGD)
pipeline.add_model(model2a)
pipeline.add_model(model2b)
pipeline.add_model(model2c)
pipeline.add_model(model2d)

############################################################


def compare_link(l1, l2):
    numpy.testing.assert_equal(l1.W.data, l2.W.data)
    numpy.testing.assert_equal(l1.b.data, l2.b.data)


def compare_links():
    compare_link(model1.l1, model2a.l1)
    compare_link(model1.l2, model2b.l2)
    compare_link(model1.l3, model2c.l3)
    compare_link(model1.l4, model2d.l4)

############################################################

model2a.l1.W.copydata(model1.l1.W)
model2a.l1.b.copydata(model1.l1.b)

model2b.l2.W.copydata(model1.l2.W)
model2b.l2.b.copydata(model1.l2.b)

model2c.l3.W.copydata(model1.l3.W)
model2c.l3.b.copydata(model1.l3.b)

model2d.l4.W.copydata(model1.l4.W)
model2d.l4.b.copydata(model1.l4.b)

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


print("#################### pipeline model ####################")
if True:
    pipeline.setup()

    for loop in range(0, num_loop):
        pipeline.run(x2, label)
        # pipeline.run_sync(x2, label)

    pipeline.finalize()


print("########## check variables ##########")

compare_links()
