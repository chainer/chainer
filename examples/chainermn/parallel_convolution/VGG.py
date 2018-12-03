from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import chainermn.functions
import numpy as np


"""
This example is ported from Chainer official VGG16 example.
https://github.com/chainer/chainer/blob/master/examples/cifar/models/VGG.py
"""


class ParallelConvolution2D(chainer.links.Convolution2D):
    def __init__(self, comm, in_channels, out_channels, *args, **kwargs):
        self.comm = comm
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(ParallelConvolution2D, self).__init__(
            self._in_channel_size, self._out_channel_size, *args, **kwargs)

    def _channel_size(self, n_channel):
        # Return the size of the corresponding channels.
        n_proc = self.comm.size
        i_proc = self.comm.rank
        return n_channel // n_proc + (1 if i_proc < n_channel % n_proc else 0)

    @property
    def _in_channel_size(self):
        return self._channel_size(self.in_channels)

    @property
    def _out_channel_size(self):
        return self._channel_size(self.out_channels)

    @property
    def _channel_indices(self):
        # Return the indices of the corresponding channel.
        indices = np.arange(self.in_channels)
        indices = indices[indices % self.comm.size == 0] + self.comm.rank
        return [i for i in indices if i < self.in_channels]

    def __call__(self, x):
        x = x[:, self._channel_indices, :, :]
        y = super(ParallelConvolution2D, self).__call__(x)
        ys = chainermn.functions.allgather(self.comm, y)
        return F.concat(ys, axis=1)


class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    The convolution performs as either single-process or model-parallel
    depending on the number of input channels.

    Args:
        comm: ChainerMN communicator.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, comm, in_channels, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            if comm.size <= in_channels:
                self.conv = ParallelConvolution2D(comm,
                                                  in_channels,
                                                  out_channels,
                                                  ksize,
                                                  pad=pad,
                                                  nobias=True)
            else:
                self.conv = chainer.links.Convolution2D(in_channels,
                                                        out_channels,
                                                        ksize,
                                                        pad=pad,
                                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(h)


class VGG(chainer.Chain):

    """A VGG-style network for very small images.

    This model implementation is ported from Chainer official example:
    https://github.com/chainer/chainer/blob/master/examples/cifar/models/VGG.py

    Args:
        comm: ChainerMN communicator.
        class_labels (int): The number of class labels.

    """

    def __init__(self, comm, class_labels=10):
        super(VGG, self).__init__()
        self.comm = comm

        with self.init_scope():
            self.block1_1 = Block(comm, 3,   64,  3)
            self.block1_2 = Block(comm, 64,  64,  3)
            self.block2_1 = Block(comm, 64,  128, 3)
            self.block2_2 = Block(comm, 128, 128, 3)
            self.block3_1 = Block(comm, 128, 256, 3)
            self.block3_2 = Block(comm, 256, 256, 3)
            self.block3_3 = Block(comm, 256, 256, 3)
            self.block4_1 = Block(comm, 256, 512, 3)
            self.block4_2 = Block(comm, 512, 512, 3)
            self.block4_3 = Block(comm, 512, 512, 3)
            self.block5_1 = Block(comm, 512, 512, 3)
            self.block5_2 = Block(comm, 512, 512, 3)
            self.block5_3 = Block(comm, 512, 512, 3)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        # 64 channel blocks:
        h = self.block1_1(x)
        h = F.dropout(h, ratio=0.3)
        h = self.block1_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 128 channel blocks:
        h = self.block2_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block2_2(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 256 channel blocks:
        h = self.block3_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block3_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block4_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block4_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        # 512 channel blocks:
        h = self.block5_1(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_2(h)
        h = F.dropout(h, ratio=0.4)
        h = self.block5_3(h)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        h = self.fc2(h)

        return h
