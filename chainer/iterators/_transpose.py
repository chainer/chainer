import six


def transpose(data):
    if isinstance(data[0], tuple):
        return tuple([d[i] for d in data]
                     for i in six.moves.range(len(data[0])))
    elif isinstance(data[0], dict):
        return {k: [d[k] for d in data] for k in data[0].keys()}
    else:
        return data
