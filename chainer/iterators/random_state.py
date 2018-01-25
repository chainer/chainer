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


def derive_random_state():
    """Derive a new PRNG state from the global one.

    The global state will be put one step forward.

    It is likely that the derived state and the global one are far distant on
    the PRNG state-transition sequence.  We need to rely on this empirical way,
    as numpy doesn't implement `random.jumpahead` like functionality, which
    allows making a jump on the state-transition sequence with arbitrary
    length.

    To support 32-bit platform and numpy < 1.11, the value is taken in
    a verbose manner.

    Returns:
        A :class:`numpy.random.RandomState` instance.

    """
    seed = numpy.asscalar(
        numpy.random.randint(-(1 << 31), 1 << 31, 1).astype('uint32'))
    return numpy.random.RandomState(seed)


def create_random_states(num):
    return [derive_random_state() for _ in range(num)]


@contextlib.contextmanager
def set_random_state(random_state):
    _random_state.value = random_state
    yield
    _random_state.value = None
