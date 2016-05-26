import chainer
import chainer.functions as F
import chainer.links as L


class VGG(chainer.Chain):

    '''VGG(network A) with dropout removed.

    '''

    insize = 224
    batchsize = 64
    in_channels = 3

    def __init__(self, batchsize, use_cudnn):
        super(VGG, self).__init__(
            conv1=L.Convolution2D(3, 64, 3, use_cudnn=use_cudnn),
            conv2=L.Convolution2D(64, 256, 3, use_cudnn=use_cudnn),
            conv3=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv4=L.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv5=L.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv6=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv7=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv8=L.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            fc6=L.Linear(25088, 4096),
            fc7=L.Linear(4096, 4096),
            fc8=L.Linear(4096, 1000)
        )
        self.use_cudnn = use_cudnn
        if batchsize is not None:
            self.batchsize = batchsize

    def __call__(self, x):
        h = F.relu(self.conv1(x), self.use_cudnn)
        h = F.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = F.relu(self.conv2(h), self.use_cudnn)
        h = F.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = F.relu(self.conv3(h), self.use_cudnn)
        h = F.relu(self.conv4(h), self.use_cudnn)
        h = F.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = F.relu(self.conv5(h), self.use_cudnn)
        h = F.relu(self.conv6(h), self.use_cudnn)
        h = F.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = F.relu(self.conv7(h), self.use_cudnn)
        h = F.relu(self.conv8(h), self.use_cudnn)
        h = F.max_pooling_2d(h, 2, stride=2, use_cudnn=self.use_cudnn)
        h = self.fc6(h)
        h = self.fc7(h)
        return self.fc8(h)
