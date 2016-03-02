import chainer
import chainer.functions as F


class Conv(chainer.FunctionSet):

    batchsize = None
    in_channels = None
    insize = None
    out_channels = None
    kernel_size = None

    def __init__(self, batchsize, use_cudnn):
        super(Conv, self).__init__(
            conv=F.Convolution2D(self.in_channels,
                                 self.out_channels,
                                 self.kernel_size, stride=1,
                                 use_cudnn=use_cudnn)
        )
        self.use_cudnn = use_cudnn
        if batchsize is not None:
            self.batchsize = batchsize

    def forward(self, x_data, train=True):
        x = chainer.Variable(x_data, volatile=not train)
        return self.conv(x)


class Conv1(Conv):

    '''Single-convolution architecture

    '''

    batchsize = 128
    in_channels = 3
    insize = 128
    out_channels = 96
    kernel_size = 11


class Conv2(Conv):

    '''Single-convolution architecture

    '''

    batchsize = 128
    in_channels = 64
    insize = 64
    out_channels = 128
    kernel_size = 9


class Conv3(Conv):

    '''Single-convolution architecture

    '''

    batchsize = 128
    in_channels = 128
    insize = 32
    out_channels = 128
    kernel_size = 9


class Conv4(Conv):

    '''Single-convolution architecture

    '''

    batchsize = 128
    in_channels = 128
    insize = 16
    out_channels = 128
    kernel_size = 7


class Conv5(Conv):

    '''Single-convolution architecture

    '''

    batchsize = 128
    in_channels = 384
    insize = 13
    out_channels = 384
    kernel_size = 3
