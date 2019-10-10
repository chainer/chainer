import chainer


def get_device(gpu_id=None, use_chainerx=False):
    """Get device object

    Currently in Chainer, we have 3 officially-supported backends
    (numpy, cupy, chainerx) and 2 devices (CPU and NVIDIA GPUs).
    Also, ChainerX has its own backend system, so there are 4 combinations
    (numpy, cupy, chainerx+native, chainerx+cuda). This utility function
    is a boilerplate to get device object in ChainerMN contexts.
    """
    if gpu_id:
        # We need to set GPU id every time we call to_device(),
        # because each test
        if use_chainerx:
            device = 'cuda:{}'.format(gpu_id)
        else:
            # cupy
            device = '@cupy:{}'.format(gpu_id)
    else:
        if use_chainerx:
            device = 'native:0'
        else:
            device = -1

    return chainer.get_device(device)
