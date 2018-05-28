import chainer
import chainer.functions as F
import chainer.links as L
from chainer import graph_summary


class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, out_channels, ksize, pad=1):
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        with graph_summary.graph([x], self.name) as g:
            h = self.conv(x)
            h = self.bn(h)
            y = F.relu(h)
            g.set_output([y])
        return y


class VGG(chainer.Chain):

    """A VGG-style network for very small images.

    This model is based on the VGG-style model from
    http://torch.ch/blog/2015/07/30/cifar.html
    which is based on the network architecture from the paper:
    https://arxiv.org/pdf/1409.1556v6.pdf

    This model is intended to be used with either RGB or greyscale input
    images that are of size 32x32 pixels, such as those in the CIFAR10
    and CIFAR100 datasets.

    On CIFAR10, it achieves approximately 89% accuracy on the test set with
    no data augmentation.

    On CIFAR100, it achieves approximately 63% accuracy on the test set with
    no data augmentation.

    Args:
        class_labels (int): The number of class labels.

    """

    def __init__(self, class_labels=10):
        super(VGG, self).__init__()
        with self.init_scope():
            self.block1_1 = Block(64, 3)
            self.block1_2 = Block(64, 3)
            self.block2_1 = Block(128, 3)
            self.block2_2 = Block(128, 3)
            self.block3_1 = Block(256, 3)
            self.block3_2 = Block(256, 3)
            self.block3_3 = Block(256, 3)
            self.block4_1 = Block(512, 3)
            self.block4_2 = Block(512, 3)
            self.block4_3 = Block(512, 3)
            self.block5_1 = Block(512, 3)
            self.block5_2 = Block(512, 3)
            self.block5_3 = Block(512, 3)
            self.fc1 = L.Linear(None, 512, nobias=True)
            self.bn_fc1 = L.BatchNormalization(512)
            self.fc2 = L.Linear(None, class_labels, nobias=True)

    def __call__(self, x):
        import threading
        current_thread = threading.current_thread()
        outer_context = current_thread.__dict__['graph_context']
        assert len(outer_context.subgraph_map) == 0

        # 64 channel blocks:
        with graph_summary.graph([x], 'vgg_block1') as g:
            h = self.block1_1(x)
            h = F.dropout(h, ratio=0.3)
            h = self.block1_2(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
            g.set_output([h])

        # 128 channel blocks:
        with graph_summary.graph([h], 'vgg_block2') as g:
            h = self.block2_1(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block2_2(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
            g.set_output([h])

        # 256 channel blocks:
        with graph_summary.graph([h], 'vgg_block3') as g:
            h = self.block3_1(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block3_2(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block3_3(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
            g.set_output([h])

        # 512 channel blocks:
        with graph_summary.graph([h], 'vgg_block4') as g:
            h = self.block4_1(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block4_2(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block4_3(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
            g.set_output([h])

        # 512 channel blocks:
        with graph_summary.graph([h], 'vgg_block5') as g:
            h = self.block5_1(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block5_2(h)
            h = F.dropout(h, ratio=0.4)
            h = self.block5_3(h)
            h = F.max_pooling_2d(h, ksize=2, stride=2)
            g.set_output([h])

        h = F.dropout(h, ratio=0.5)
        h = self.fc1(h)
        h = self.bn_fc1(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=0.5)
        return self.fc2(h)
