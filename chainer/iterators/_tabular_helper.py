import six


def apply(convert, data):
    if isinstance(data, tuple):
        return convert(*data)
    elif isinstance(data, dict):
        return convert(**data)


def transpose(data):
    if isinstance(data[0], tuple):
        return tuple([d[i] for d in data]
                     for i in six.moves.range(len(data[0])))
    elif isinstance(data[0], dict):
        return {k: [d[k] for d in data] for k in data[0].keys()}
