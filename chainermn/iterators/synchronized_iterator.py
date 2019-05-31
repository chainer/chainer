import chainer
import numpy


class _SynchronizedIterator(chainer.dataset.iterator.Iterator):

    def __init__(self, actual_iterator, communicator):
        if not hasattr(actual_iterator, 'order_sampler'):
            raise ValueError('actual_iterator must have order_sampler')
        else:
            super(_SynchronizedIterator, self).__setattr__(
                'actual_iterator', actual_iterator)

        # Synchronize random seed.
        self.communicator = communicator
        if self.communicator.rank == 0:
            seed = numpy.random.randint(0, 2 ** 32 - 1)
        else:
            seed = None
        seed = self.communicator.bcast_obj(seed, root=0)

        # Random number generator for iterator.
        rng = numpy.random.RandomState(seed)
        self.actual_iterator.order_sampler = \
            chainer.iterators.ShuffleOrderSampler(rng)
        self.actual_iterator.reset()

    def __getattr__(self, attr_name):
        return getattr(self.actual_iterator, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_iterator, attr_name, value)

    def __next__(self):
        return self.actual_iterator.__next__()

    def serialize(self, serializer):
        self.actual_iterator.serialize(serializer)


def create_synchronized_iterator(actual_iterator, communicator):
    """Create a synchronized iterator from a Chainer iterator.

    This iterator shares the same batches on multiple processes,
    using the same random number generators to maintain the order of batch
    shuffling same.

    Here is an example situation.
    When we train a sequence-to-sequence model, where the encoder and
    the decoder is located on two different processes, we want to share
    the same batches on each process, thus inputs for the encoder and
    output teacher signals for the decoder become consistent.

    In order to use the synchronized iterator, first create the iterator
    from Chainer iterator and ChainerMN communicator::

        iterator = chainermn.iterators.create_synchronized_iterator(
            chainer.iterators.SerialIterator(
                dataset, batch_size, shuffle=True),
            communicator)

    Then you can use it as the ordinary Chainer iterator::

        updater = chainer.training.StandardUpdater(iterator, optimizer)
        trainer = training.Trainer(updater)
        trainer.run()

    The resulting iterator shares the same shuffling order among processes
    in the specified communicator.

    Args:
        actual_iterator: Chainer iterator
            (e.g., ``chainer.iterators.SerialIterator``).
        communicator: ChainerMN communicator.

    Returns:
        The synchronized iterator based on ``actual_iterator``.
    """
    chainer.utils.experimental(
        'chainermn.iterators.create_synchronized_iterator')

    return _SynchronizedIterator(actual_iterator, communicator)
