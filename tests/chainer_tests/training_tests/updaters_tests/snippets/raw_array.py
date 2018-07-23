import numpy

import chainer
from chainer import testing
import chainer.training.updaters.multiprocess_parallel_updater as mpu


class SimpleNetRawArray(chainer.Chain):

    def __init__(self):
        super(SimpleNetRawArray, self).__init__()
        with self.init_scope():
            self.conv = chainer.links.Convolution2D(2, 2, 3)
            self.fc = chainer.links.Linear(18, 2)

        self.train = True
        self.call_called = 0

    def clear(self):
        self.loss = None
        self.accuracy = None

    def forward(self, x, t):
        assert not isinstance(x, chainer.Variable)
        assert not isinstance(t, chainer.Variable)

        self.call_called += 1

        h = chainer.functions.relu(self.conv(x))
        y = self.fc(h)

        self.loss = chainer.functions.softmax_cross_entropy(y, t)
        self.accuracy = chainer.functions.accuracy(y, t)

        return self.loss


def test():
    model = SimpleNetRawArray()
    dataset = [((numpy.ones((2, 5, 5)) * i).astype(numpy.float32),
                numpy.int32(0)) for i in range(100)]

    batch_size = 5
    devices = (0,)
    iters = [chainer.iterators.SerialIterator(i, batch_size) for i in
             chainer.datasets.split_dataset_n_random(
                 dataset, len(devices))]
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)

    with testing.assert_warns(UserWarning):
        updater = mpu.MultiprocessParallelUpdater(
            iters, optimizer, devices=devices)
    updater.update()

    assert model.call_called == 1


if __name__ == '__main__':
    test()


# This snippet is not a test code.
# testing.run_module(__name__, __file__)
