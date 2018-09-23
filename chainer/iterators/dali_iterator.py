from __future__ import division

from chainer.dataset import iterator
from chainer import utils


class DaliIterator(iterator.Iterator):

    """(Experimental) Iterator for DALI pipeline.

    Args:
        pipeline: DALI pipeline.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.

    """

    def __init__(self, pipeline, repeat=True):
        utils.experimental('DaliIterator')
        self.pipeline = pipeline
        self._repeat = repeat
        self._is_build = False
        self.epoch_size = 1  # dummy
        self.reset()

    def __next__(self):
        if not self._is_build:
            self.pipeline.build()
            self._is_build = True
            self.epoch_size = tuple(self.pipeline.epoch_size().values())[0]

        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = self.epoch_size
        if i_end >= N:
            if self._repeat:
                self.current_position = i_end - N
            else:
                self.current_position = 0
            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.current_position = i_end
            self.is_new_epoch = False

        return self.pipeline.run()

    next = __next__

    @property
    def batch_size(self):
        return self.pipeline.batch_size

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / self.epoch_size

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / self.epoch_size
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

    def reset(self):
        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

    @property
    def repeat(self):
        return self._repeat
