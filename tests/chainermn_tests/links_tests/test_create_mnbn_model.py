import unittest

import chainer
import chainer.testing
import chainer.testing.attr
import chainermn


class BnChain(chainer.Chain):

    def __init__(self, size):
        super(BnChain, self).__init__()
        with self.init_scope():
            self.conv = chainer.links.Convolution2D(
                None, size, 1, 1, 1, nobias=True)
            self.bn = chainer.links.BatchNormalization(size)

    def forward(self, x):
        return chainer.functions.relu(self.bn(self.conv(x)))


class BnChainList(chainer.ChainList):

    def __init__(self, size):
        super(BnChainList, self).__init__(
            chainer.links.Convolution2D(
                None, size, 1, 1, 1, nobias=True),
            chainer.links.BatchNormalization(size),
        )

    def forward(self, x):
        return chainer.functions.relu(self[1](self[0](x)))


class TestCreateMnBnModel(unittest.TestCase):

    def setUp(self):
        self.communicator = chainermn.create_communicator('naive')

    def check_create_mnbn_model_chain(self, gpu):
        model = BnChain(3)
        mnbn_model = chainermn.links.create_mnbn_model(model,
                                                       self.communicator)
        self.assertTrue(isinstance(mnbn_model.conv,
                                   chainer.links.Convolution2D))
        self.assertTrue(
            isinstance(mnbn_model.bn,
                       chainermn.links.MultiNodeBatchNormalization))
        if gpu:
            device_id = self.communicator.intra_rank
            mnbn_model.to_gpu(device=device_id)
        else:
            device_id = -1
        with chainer.using_device(mnbn_model.device):
            x = mnbn_model.xp.zeros((1, 1, 1, 1))
            mnbn_model(x)

    def check_create_mnbn_model_chain_list(self, gpu):
        model = BnChainList(3)
        mnbn_model = chainermn.links.create_mnbn_model(model,
                                                       self.communicator)
        self.assertTrue(isinstance(mnbn_model[0],
                                   chainer.links.Convolution2D))
        self.assertTrue(
            isinstance(mnbn_model[1],
                       chainermn.links.MultiNodeBatchNormalization))
        if gpu:
            device_id = self.communicator.intra_rank
            mnbn_model.to_gpu(device=device_id)
        else:
            device_id = -1
        with chainer.using_device(mnbn_model.device):
            x = mnbn_model.xp.zeros((1, 1, 1, 1))
            mnbn_model(x)

    def check_create_mnbn_model_sequential(self, gpu):
        size = 3
        model = chainer.Sequential(
            chainer.links.Convolution2D(
                None, size, 1, 1, 1, nobias=True),
            chainer.links.BatchNormalization(size),
            chainer.functions.relu
        )
        mnbn_model = chainermn.links.create_mnbn_model(model,
                                                       self.communicator)

        if gpu:
            device_id = self.communicator.intra_rank
            mnbn_model.to_gpu(device=device_id)
        else:
            device_id = -1
        with chainer.using_device(mnbn_model.device):
            x = mnbn_model.xp.zeros((1, 1, 1, 1))
            mnbn_model(x)

    def test_create_mnbn_model_chain_cpu(self):
        self.check_create_mnbn_model_chain(gpu=False)

    def test_create_mnbn_model_chain_list_cpu(self):
        self.check_create_mnbn_model_chain_list(gpu=False)

    def test_create_mnbn_model_sequential_cpu(self):
        self.check_create_mnbn_model_sequential(gpu=False)

    @chainer.testing.attr.gpu
    def test_create_mnbn_model_chain_gpu(self):
        self.check_create_mnbn_model_chain(gpu=True)

    @chainer.testing.attr.gpu
    def test_create_mnbn_model_chain_list_gpu(self):
        self.check_create_mnbn_model_chain_list(gpu=True)

    @chainer.testing.attr.gpu
    def test_create_mnbn_model_sequential_gpu(self):
        self.check_create_mnbn_model_sequential(gpu=True)
