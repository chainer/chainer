import contextlib
import threading

import numpy


_random_state = threading.local()
_random_state.value = None


def get_random_state():
    """Get a :class:`~numpy.random.RandomState` during iteration.

    The instance obtained is unique to each index on a batch.  You can
    employ the state, for example, to perform random data augmentation.
    No manual initialization is required even if you are using
    :class:`~chainer.iterators.MultiprocessIterator` or
    :class:`~chainer.iterators.MultithreadIterator`.

    The state is deterministically determined by the initial state of
    ``numpy.random``. Therefore you can reproduce the same result
    if you set the same seed by :func:`numpy.random.seed` at the program
    startup.

    The states remain unchanged when an iterator is reset. Internal states
    during prefetch are silently discarded.

    Returns:
        The :class:`numpy.random.RandomState` instance bound to the batch
        index, if a :class:`~chainer.dataset.Iterator` instance is fetching
        data; otherwise ``None``.

    """
    state = _random_state.value
    if not isinstance(state, numpy.random.RandomState):
        state = state()
    return state


@contextlib.contextmanager
def set_random_state(random_state):
    """Create a context for :meth:`get_random_state` with a given PRNG state.

    This function shall be called by iterator implementation each time
    it yields a batch.

    Args:
        random_state (numpy.random.RandomState or callable): a PRNG state.
            If it is callable, it must return a :numpy.random.RandomState:
            instance when called with no argument.
    """
    _random_state.value = random_state
    yield
    _random_state.value = None
