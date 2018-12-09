import chainer
import chainer.functions as F
import chainer.links as L


class ResidualBlock(chainer.Chain):
    def __init__(self, filter_size, dilation,
                 residual_channels, dilated_channels, skip_channels):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(
                residual_channels, dilated_channels,
                ksize=(filter_size, 1),
                pad=(dilation * (filter_size - 1), 0), dilate=(dilation, 1))
            self.res = L.Convolution2D(
                dilated_channels // 2, residual_channels, 1)
            self.skip = L.Convolution2D(
                dilated_channels // 2, skip_channels, 1)

        self.filter_size = filter_size
        self.dilation = dilation
        self.residual_channels = residual_channels

    def __call__(self, x, condition):
        length = x.shape[2]
        h = self.conv(x)
        h = h[:, :, :length]  # crop
        h += condition
        tanh_z, sig_z = F.split_axis(h, 2, axis=1)
        z = F.tanh(tanh_z) * F.sigmoid(sig_z)
        if x.shape[2] == z.shape[2]:
            residual = self.res(z) + x
        else:
            residual = self.res(z) + x[:, :, -1:]  # crop
        skip_conenection = self.skip(z)
        return residual, skip_conenection

    def initialize(self, n):
        self.queue = chainer.Variable(self.xp.zeros((
            n, self.residual_channels,
            self.dilation * (self.filter_size - 1) + 1, 1),
            dtype=self.xp.float32))
        self.conv.pad = (0, 0)

    def pop(self, condition):
        return self(self.queue, condition)

    def push(self, x):
        self.queue = F.concat((self.queue[:, :, 1:], x), axis=2)


class ResidualNet(chainer.ChainList):
    def __init__(self, n_loop, n_layer, filter_size,
                 residual_channels, dilated_channels, skip_channels):
        super(ResidualNet, self).__init__()
        dilations = [2 ** i for i in range(n_layer)] * n_loop
        for dilation in dilations:
            self.add_link(ResidualBlock(
                filter_size, dilation,
                residual_channels, dilated_channels, skip_channels))

    def __call__(self, x, conditions):
        for i, (func, cond) in enumerate(zip(self.children(), conditions)):
            x, skip = func(x, cond)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        return skip_connections

    def initialize(self, n):
        for block in self.children():
            block.initialize(n)

    def generate(self, x, conditions):
        for i, (func, cond) in enumerate(zip(self.children(), conditions)):
            func.push(x)
            x, skip = func.pop(cond)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        return skip_connections
