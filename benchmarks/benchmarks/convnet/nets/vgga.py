import chainer
import chainer.functions as F
import chainer.links as L


class vgga(chainer.Chain):
    insize = 224

    def __init__(self):
        super(vgga, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(  3,  64, 3, stride=1, pad=1)
            self.conv2 = L.Convolution2D( 64, 128, 3, stride=1, pad=1)
            self.conv3 = L.Convolution2D(128, 256, 3, stride=1, pad=1)
            self.conv4 = L.Convolution2D(256, 256, 3, stride=1, pad=1)
            self.conv5 = L.Convolution2D(256, 512, 3, stride=1, pad=1)
            self.conv6 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv7 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.conv8 = L.Convolution2D(512, 512, 3, stride=1, pad=1)
            self.fc6 = L.Linear(512 * 7 * 7, 4096)
            self.fc7 = L.Linear(4096, 4096)
            self.fc8 = L.Linear(4096, 1000)

    def forward(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2, stride=2)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2, stride=2)
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(F.relu(self.conv8(h)), 2, stride=2)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        return self.fc8(h)
