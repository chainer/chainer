import chainer
import chainer.functions as F
import chainer.links as L


class GoogLeNet(chainer.Chain):

    insize = 224

    def __init__(self):
        super(GoogLeNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3,  64, 7, stride=2, pad=3)
            self.conv2_reduce = L.Convolution2D(64,  64, 1)
            self.conv2 = L.Convolution2D(64, 192, 3, stride=1, pad=1)
            self.inc3a = L.Inception(192,  64,  96, 128, 16,  32,  32)
            self.inc3b = L.Inception(256, 128, 128, 192, 32,  96,  64)
            self.inc4a = L.Inception(480, 192,  96, 208, 16,  48,  64)
            self.inc4b = L.Inception(512, 160, 112, 224, 24,  64,  64)
            self.inc4c = L.Inception(512, 128, 128, 256, 24,  64,  64)
            self.inc4d = L.Inception(512, 112, 144, 288, 32,  64,  64)
            self.inc4e = L.Inception(528, 256, 160, 320, 32, 128, 128)
            self.inc5a = L.Inception(832, 256, 160, 320, 32, 128, 128)
            self.inc5b = L.Inception(832, 384, 192, 384, 48, 128, 128)
            self.loss3_fc = L.Linear(1024, 1000)

            self.loss1_conv = L.Convolution2D(512, 128, 1)
            self.loss1_fc1 = L.Linear(4 * 4 * 128, 1024)
            self.loss1_fc2 = L.Linear(1024, 1000)

            self.loss2_conv = L.Convolution2D(528, 128, 1)
            self.loss2_fc1 = L.Linear(4 * 4 * 128, 1024)
            self.loss2_fc2 = L.Linear(1024, 1000)

    def forward(self, x):
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

        if chainer.config.train:
            out1 = F.average_pooling_2d(h, 5, stride=3)
            out1 = F.relu(self.loss1_conv(out1))
            out1 = F.relu(self.loss1_fc1(out1))
            out1 = self.loss1_fc2(out1)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        if chainer.config.train:
            out2 = F.average_pooling_2d(h, 5, stride=3)
            out2 = F.relu(self.loss2_conv(out2))
            out2 = F.relu(self.loss2_fc1(out2))
            out2 = self.loss2_fc2(out2)

        h = self.inc4e(h)
        h = F.max_pooling_2d(h, 3, stride=2)
        h = self.inc5a(h)
        h = self.inc5b(h)

        h = F.dropout(F.average_pooling_2d(h, 7, stride=1), 0.4)
        out3 = self.loss3_fc(h)
        return out1, out2, out3
