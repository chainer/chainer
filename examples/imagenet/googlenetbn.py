import chainer
import chainer.functions as F
import chainer.links as L


class GoogLeNetBN(chainer.Chain):

    """New GoogLeNet of BatchNormalization version."""

    insize = 224

    def __init__(self):
        super(GoogLeNetBN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, 64, 7, stride=2, pad=3, nobias=True)
            self.norm1 = L.BatchNormalization(64)
            self.conv2 = L.Convolution2D(None, 192, 3, pad=1, nobias=True)
            self.norm2 = L.BatchNormalization(192)
            self.inc3a = L.InceptionBN(
                None, 64, 64, 64, 64, 96, 'avg', 32)
            self.inc3b = L.InceptionBN(
                None, 64, 64, 96, 64, 96, 'avg', 64)
            self.inc3c = L.InceptionBN(
                None, 0, 128, 160, 64, 96, 'max', stride=2)
            self.inc4a = L.InceptionBN(
                None, 224, 64, 96, 96, 128, 'avg', 128)
            self.inc4b = L.InceptionBN(
                None, 192, 96, 128, 96, 128, 'avg', 128)
            self.inc4c = L.InceptionBN(
                None, 160, 128, 160, 128, 160, 'avg', 128)
            self.inc4d = L.InceptionBN(
                None, 96, 128, 192, 160, 192, 'avg', 128)
            self.inc4e = L.InceptionBN(
                None, 0, 128, 192, 192, 256, 'max', stride=2)
            self.inc5a = L.InceptionBN(
                None, 352, 192, 320, 160, 224, 'avg', 128)
            self.inc5b = L.InceptionBN(
                None, 352, 192, 320, 192, 224, 'max', 128)
            self.out = L.Linear(None, 1000)

            self.conva = L.Convolution2D(None, 128, 1, nobias=True)
            self.norma = L.BatchNormalization(128)
            self.lina = L.Linear(None, 1024, nobias=True)
            self.norma2 = L.BatchNormalization(1024)
            self.outa = L.Linear(None, 1000)

            self.convb = L.Convolution2D(None, 128, 1, nobias=True)
            self.normb = L.BatchNormalization(128)
            self.linb = L.Linear(None, 1024, nobias=True)
            self.normb2 = L.BatchNormalization(1024)
            self.outb = L.Linear(None, 1000)

    def forward(self, x, t):
        h = F.max_pooling_2d(
            F.relu(self.norm1(self.conv1(x))),  3, stride=2, pad=1)
        h = F.max_pooling_2d(
            F.relu(self.norm2(self.conv2(h))), 3, stride=2, pad=1)

        h = self.inc3a(h)
        h = self.inc3b(h)
        h = self.inc3c(h)
        h = self.inc4a(h)

        a = F.average_pooling_2d(h, 5, stride=3)
        a = F.relu(self.norma(self.conva(a)))
        a = F.relu(self.norma2(self.lina(a)))
        a = self.outa(a)
        loss1 = F.softmax_cross_entropy(a, t)

        h = self.inc4b(h)
        h = self.inc4c(h)
        h = self.inc4d(h)

        b = F.average_pooling_2d(h, 5, stride=3)
        b = F.relu(self.normb(self.convb(b)))
        b = F.relu(self.normb2(self.linb(b)))
        b = self.outb(b)
        loss2 = F.softmax_cross_entropy(b, t)

        h = self.inc4e(h)
        h = self.inc5a(h)
        h = F.average_pooling_2d(self.inc5b(h), 7)
        h = self.out(h)
        loss3 = F.softmax_cross_entropy(h, t)

        loss = 0.3 * (loss1 + loss2) + loss3
        accuracy = F.accuracy(h, t)

        chainer.report({
            'loss': loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'accuracy': accuracy,
        }, self)
        return loss
