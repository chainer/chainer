import chainer
import chainer.functions as F


class VGG(chainer.FunctionSet):

    '''VGG(network A) with dropout removed.

    '''

    insize = 224
    batchsize = 64
    in_channels = 3

    def __init__(self, batchsize, use_cudnn):
        super(VGG, self).__init__(
            conv1=F.Convolution2D(3, 64, 3, use_cudnn=use_cudnn),
            conv2=F.Convolution2D(64, 256, 3, use_cudnn=use_cudnn),
            conv3=F.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv4=F.Convolution2D(256, 256, 3, pad=1, use_cudnn=use_cudnn),
            conv5=F.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv6=F.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv7=F.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv8=F.Convolution2D(512, 512, 3, pad=1, use_cudnn=use_cudnn),
            fc6=F.Linear(25088, 4096),
            fc7=F.Linear(4096, 4096),
            fc8=F.Linear(4096, 1000)
        )
        self.use_cudnn = use_cudnn
        if batchsize is not None:
            self.batchsize = batchsize

    def forward(self, x_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)

        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2, stride=2)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2, stride=2)
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(F.relu(self.conv8(h)), 2, stride=2)
        h = self.fc6(h)
        h = self.fc7(h)
        return self.fc8(h)
