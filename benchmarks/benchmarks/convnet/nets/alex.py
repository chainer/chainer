import chainer
import chainer.functions as F
import chainer.links as L


class Alex(chainer.Chain):
    insize = 224

    def __init__(self):
        super(Alex, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(3,  64, 11, stride=4, pad=2)
            self.conv2 = L.Convolution2D(64, 192,  5, pad=2)
            self.conv3 = L.Convolution2D(192, 384,  3, pad=1)
            self.conv4 = L.Convolution2D(384, 256,  3, pad=1)
            self.conv5 = L.Convolution2D(256, 256,  3, pad=1)
            self.fc6 = L.Linear(256 * 6 * 6, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def forward(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        return self.fc8(h)
