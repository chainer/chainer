import multiprocessing
import sys

import numpy

import chainer
from chainer.backends import cuda
import chainer.training.updaters.multiprocess_parallel_updater as mpu


class SimpleNetChild(chainer.Chain):

    def __init__(self):
        super(SimpleNetChild, self).__init__()
        with self.init_scope():
            self.c1 = chainer.links.Convolution2D(2, 2, 3)
            self.fc = chainer.links.Linear(18, 2)

    def clear(self):
        self.loss = None

    def forward(self, x, t):

        h = chainer.functions.relu(self.c1(x))
        y = self.fc(h)

        self.loss = chainer.functions.softmax_cross_entropy(y, t)

        return self.loss


def test():
    model = SimpleNetChild()
    dataset = [((numpy.ones((2, 5, 5)) * i).astype(numpy.float32),
                numpy.int32(0)) for i in range(100)]

    batch_size = 5
    devices = tuple([chainer.get_device(d) for d in sys.argv[1].split(',')])
    iters = [chainer.iterators.SerialIterator(i, batch_size) for i in
             chainer.datasets.split_dataset_n_random(
                 dataset, len(devices))]
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)

    # Initialize CUDA context.
    cuda.cupy.cuda.runtime.runtimeGetVersion()

    try:
        mpu.MultiprocessParallelUpdater(iters, optimizer, devices=devices)
    except RuntimeError as e:
        if sys.argv[2] == 'fork':
            assert 'CUDA context' in str(e)
            return

    updater = mpu.MultiprocessParallelUpdater(
        iters, optimizer, devices=devices)
    trainer = chainer.training.Trainer(updater, (1, 'epoch'), '/tmp')
    trainer.run()
    # assert False


if __name__ == '__main__':
    multiprocessing.set_start_method(sys.argv[2])
    test()


# This snippet is not a test code.
# testing.run_module(__name__, __file__)
