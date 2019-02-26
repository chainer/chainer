import unittest

import chainer
import chainermn


class BnChain(chainer.Chain):

    def __init__(self, size):
        super(BnChain, self).__init__()
        with self.init_scope():
            self.conv = chainer.links.Convolution2D(
                None, size, 1, 1, 1, nobias=True)
            self.bn = chainer.links.BatchNormalization(size)

    def forward(self, x):
        return self.bn(self.conv(x))


class BnChainList(chainer.ChainList):

    def __init__(self, size):
        super(BnChainList, self).__init__(
            chainer.links.Convolution2D(
                None, size, 1, 1, 1, nobias=True),
            chainer.links.BatchNormalization(size),
        )

    def forward(self, x):
        return self[1](self[0](x))


class TestCreateMnBnModel(unittest.TestCase):

    def setUp(self):
        self.communicator = chainermn.create_communicator('naive')

    def test_create_mnbn_model_chain(self):
        model = BnChain(3)
        mnbn_model = chainermn.links.create_mnbn_model(model,
                                                       self.communicator)
        self.assertTrue(isinstance(mnbn_model.conv,
                                   chainer.links.Convolution2D))
        self.assertTrue(
            isinstance(mnbn_model.bn,
                       chainermn.links.MultiNodeBatchNormalization))

    def test_create_mnbn_model_chain_list(self):
        model = BnChainList(3)
        mnbn_model = chainermn.links.create_mnbn_model(model,
                                                       self.communicator)
        self.assertTrue(isinstance(mnbn_model[0],
                                   chainer.links.Convolution2D))
        self.assertTrue(
            isinstance(mnbn_model[1],
                       chainermn.links.MultiNodeBatchNormalization))

    def test_create_mnbn_model_sequential(self):
        size = 3
        model = chainer.Sequential(
            chainer.links.Convolution2D(
                None, size, 1, 1, 1, nobias=True),
            chainer.links.BatchNormalization(size),
        )
        mnbn_model = chainermn.links.create_mnbn_model(model,
                                                       self.communicator)
        self.assertTrue(isinstance(mnbn_model[0],
                                   chainer.links.Convolution2D))
        self.assertTrue(
            isinstance(mnbn_model[1],
                       chainermn.links.MultiNodeBatchNormalization))
