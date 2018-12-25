import sys

import numpy

import chainer
from chainer.training import trainer
import chainer.training.updaters.multiprocess_parallel_updater as mpu


class SimpleNetChild(chainer.Chain):

    def __init__(self):
        super(SimpleNetChild, self).__init__()
        with self.init_scope():
            self.conv = chainer.links.Convolution2D(2, 2, 3)

    def forward(self, x):

        h = chainer.functions.relu(self.conv(x))

        chainer.reporter.report({
            'h_max': chainer.functions.math.minmax.max(h)}, self)

        return h


class SimpleNetChildReporter(chainer.Chain):

    def __init__(self):
        super(SimpleNetChildReporter, self).__init__()
        with self.init_scope():
            self.c1 = SimpleNetChild()
            self.fc = chainer.links.Linear(18, 2)
        self.call_called = 0

    def clear(self):
        self.loss = None

    def forward(self, x, t):

        self.call_called += 1

        h = chainer.functions.relu(self.c1(x))
        y = self.fc(h)

        self.loss = chainer.functions.softmax_cross_entropy(y, t)
        chainer.reporter.report({'loss': self.loss}, self)

        return self.loss


if __name__ == '__main__':
    model = SimpleNetChildReporter()
    dataset = [(numpy.full((2, 5, 5), i, numpy.float32),
                numpy.int32(0)) for i in range(100)]

    batch_size = 5
    devices = tuple([int(x) for x in sys.argv[1].split(',')])
    iters = [chainer.iterators.SerialIterator(i, batch_size) for i in
             chainer.datasets.split_dataset_n_random(
                 dataset, len(devices))]
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    updater = mpu.MultiprocessParallelUpdater(
        iters, optimizer, devices=devices)
    trainer = trainer.Trainer(updater, (1, 'iteration'), '/tmp')
    trainer.run()
    assert model.call_called == 1


# This snippet is not a test code.
# testing.run_module(__name__, __file__)
