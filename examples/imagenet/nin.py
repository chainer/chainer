import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L


class NIN(chainer.Chain):

    """Network-in-Network example model."""

    insize = 227

    def __init__(self):
        super(NIN, self).__init__()
        conv_init = I.HeNormal()  # MSRA scaling

        with self.init_scope():
            self.mlpconv1 = L.MLPConvolution2D(
                None, (96, 96, 96), 11, stride=4, conv_init=conv_init)
            self.mlpconv2 = L.MLPConvolution2D(
                None, (256, 256, 256), 5, pad=2, conv_init=conv_init)
            self.mlpconv3 = L.MLPConvolution2D(
                None, (384, 384, 384), 3, pad=1, conv_init=conv_init)
            self.mlpconv4 = L.MLPConvolution2D(
                None, (1024, 1024, 1000), 3, pad=1, conv_init=conv_init)

    def forward(self, x, t):
        h = F.max_pooling_2d(F.relu(self.mlpconv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.mlpconv2(h)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.mlpconv3(h)), 3, stride=2)
        h = self.mlpconv4(F.dropout(h))
        h = F.reshape(F.average_pooling_2d(h, 6), (len(x), 1000))

        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss
