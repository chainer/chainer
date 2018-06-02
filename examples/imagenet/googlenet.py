import chainer
import chainer.functions as F
import chainer.links as L


class GoogLeNet(chainer.Chain):

    insize = 224

    def __init__(self):
        super(GoogLeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None,  64, 7, stride=2, pad=3)
            self.conv2_reduce = L.Convolution2D(None,  64, 1)
            self.conv2 = L.Convolution2D(None, 192, 3, stride=1, pad=1)
            self.inc3a = L.Inception(None,  64,  96, 128, 16,  32,  32)
            self.inc3b = L.Inception(None, 128, 128, 192, 32,  96,  64)
            self.inc4a = L.Inception(None, 192,  96, 208, 16,  48,  64)
            self.inc4b = L.Inception(None, 160, 112, 224, 24,  64,  64)
            self.inc4c = L.Inception(None, 128, 128, 256, 24,  64,  64)
            self.inc4d = L.Inception(None, 112, 144, 288, 32,  64,  64)
            self.inc4e = L.Inception(None, 256, 160, 320, 32, 128, 128)
            self.inc5a = L.Inception(None, 256, 160, 320, 32, 128, 128)
            self.inc5b = L.Inception(None, 384, 192, 384, 48, 128, 128)
            self.loss3_fc = L.Linear(None, 1000)

            self.loss1_conv = L.Convolution2D(None, 128, 1)
            self.loss1_fc1 = L.Linear(None, 1024)
            self.loss1_fc2 = L.Linear(None, 1000)

            self.loss2_conv = L.Convolution2D(None, 128, 1)
            self.loss2_fc1 = L.Linear(None, 1024)
            self.loss2_fc2 = L.Linear(None, 1000)

    def forward(self, x, t):
        h = F.relu(self.conv1(x))
        h = F.local_response_normalization(
            F.max_pooling_2d(h, 3, stride=2), n=5)
        h = F.relu(self.conv2_reduce(h))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(
            F.local_response_normalization(h, n=5), 3, stride=2)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc4a(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss1_conv(l))
        l = F.relu(self.loss1_fc1(l))
        l = self.loss1_fc2(l)
        loss1 = F.softmax_cross_entropy(l, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        l = F.average_pooling_2d(h, 5, stride=3)
        l = F.relu(self.loss2_conv(l))
        l = F.relu(self.loss2_fc1(l))
        l = self.loss2_fc2(l)
        loss2 = F.softmax_cross_entropy(l, t)

        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)

        h = F.average_pooling_2d(h, 7, stride=1)
        h = self.loss3_fc(F.dropout(h, 0.4))
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy
        }, self)
        return loss
