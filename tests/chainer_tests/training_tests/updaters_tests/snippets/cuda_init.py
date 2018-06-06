import numpy

import chainer
from chainer.backends import cuda
import chainer.training.updaters.multiprocess_parallel_updater as mpu


def test():
    model = chainer.Link()
    dataset = [((numpy.ones((2, 5, 5)) * i).astype(numpy.float32),
                numpy.int32(0)) for i in range(100)]

    batch_size = 5
    devices = (0,)
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
        assert 'CUDA context' in str(e)
        return

    assert False


if __name__ == '__main__':
    test()


# This snippet is not a test code.
# testing.run_module(__name__, __file__)
