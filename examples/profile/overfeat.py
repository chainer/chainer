import chainer
import chainer.functions as F


class Overfeat(chainer.FunctionSet):

    '''Overfeat(fast model) with dropout removed.

    '''

    insize = 231
    batchsize = 128
    in_channels = 3

    def __init__(self, batchsize, use_cudnn):
        super(Overfeat, self).__init__(
            conv1=F.Convolution2D(3, 96, 11, stride=4, use_cudnn=use_cudnn),
            conv2=F.Convolution2D(96, 256, 5, use_cudnn=use_cudnn),
            conv3=F.Convolution2D(256, 512, 3, pad=1, use_cudnn=use_cudnn),
            conv4=F.Convolution2D(512, 1024, 3, pad=1, use_cudnn=use_cudnn),
            conv5=F.Convolution2D(1024, 1024, 3, pad=1, use_cudnn=use_cudnn),
            fc6=F.Linear(36864, 3072),
            fc7=F.Linear(3072, 4096),
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
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=2)
        h = self.fc6(h)
        h = self.fc7(h)
        return self.fc8(h)
