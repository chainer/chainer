import chainer


def to_device(model, communicator, use_gpu, use_chx):
    """Send Chainer model to devices

    Currently in Chainer, we have 3 officially-supported backends
    (numpy, cupy, chainerx) and 2 devices (CPU and NVIDIA GPUs).
    Also, ChainerX has its own backend system, so there are 4 combinations
    (numpy, cupy, chainerx+native, chainerx+cuda). This utility function
    is a boilerplate to send Chainer model to backend devices
    in tests in test/chainermn_tests.
    """
    if use_gpu:
        # We need to set GPU id every time we call to_device(),
        # because each test
        chainer.cuda.get_device_from_id(communicator.intra_rank).use()
        if use_chx:
            device = 'cuda:{}'.format(communicator.intra_rank)
        else:
            # cupy
            device = '@cupy:{}'.format(communicator.intra_rank)
    else:
        if use_chx:
            device = 'native:0'
        else:
            device = -1

    device = chainer.get_device(device)
    model.to_device(device)
