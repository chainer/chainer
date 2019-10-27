import chainer


def get_device(device_id=None, use_chainerx=False):
    """Get device object

    Currently in Chainer, there are 3 officially-supported backends
    (numpy, cupy, and chainerx) and 2 devices (CPU and NVIDIA GPUs).
    Also, ChainerX has its own backend system, so there are 4 combinations
    (numpy, cupy, chainerx+native, chainerx+cuda). This utility function
    is a boilerplate to get device object in ChainerMN contexts.
    """
    if device_id is not None:
        if use_chainerx:
            device = 'cuda:{}'.format(device_id)
        else:
            device = '@cupy:{}'.format(device_id)
    else:
        if use_chainerx:
            device = 'native:0'
        else:
            device = '@numpy'

    return chainer.get_device(device)
