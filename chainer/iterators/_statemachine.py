import collections

import numpy


IteratorState = collections.namedtuple('IteratorState', (
    'current_position', 'epoch', 'is_new_epoch', 'order'))


def iterator_statemachine(state, batch_size, repeat, order_sampler,
                          dataset_len):
    i, epoch, _, order = state

    if not repeat and epoch > 0:
        return state, None

    indices_list = []

    n = dataset_len if order is None else len(order)
    i_end = i + batch_size
    if not repeat:
        i_end = min(i_end, n)
    is_new_epoch = False

    while i_end >= n:
        if order is None:
            indices_list.append(numpy.arange(i, n, dtype=numpy.intp))
        else:
            indices_list.append(order[i:n])

        if order is not None:
            new_order = order_sampler(order, i)
            if len(new_order) != len(order):
                raise ValueError('The size of order does not match '
                                 'the size of the previous order.')
            order = new_order

        i = 0
        i_end = i_end - n if repeat else 0
        epoch += 1
        is_new_epoch = True

    if order is None:
        indices_list.append(numpy.arange(i, i_end, dtype=numpy.intp))
    else:
        indices_list.append(order[i:i_end])

    state = IteratorState(i_end, epoch, is_new_epoch, order)
    indices = numpy.concatenate(indices_list)
    return state, indices
