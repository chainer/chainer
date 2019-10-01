import chainer
import chainer.functions as F
import chainer.links as L

from modules import ResidualNet


class UpsampleNet(chainer.ChainList):
    def __init__(self, out_layers, r_channels,
                 channels=[128, 128], upscale_factors=[16, 16]):
        super(UpsampleNet, self).__init__()
        for channel, factor in zip(channels, upscale_factors):
            self.add_link(L.Deconvolution1D(
                None, channel, factor, stride=factor, pad=0))
        for i in range(out_layers):
            self.add_link(L.Convolution1D(None, 2 * r_channels, 1))
        self.n_deconvolutions = len(channels)

    def forward(self, x):
        conditions = []
        for i, link in enumerate(self.children()):
            if i < self.n_deconvolutions:
                x = F.relu(link(x))
            else:
                conditions.append(link(x))
        return F.stack(conditions)


class WaveNet(chainer.Chain):
    def __init__(self, n_loop, n_layer, a_channels, r_channels, s_channels,
                 use_embed_tanh):
        super(WaveNet, self).__init__()
        with self.init_scope():
            self.embed = L.Convolution1D(
                a_channels, r_channels, 2, pad=1, nobias=True)
            self.resnet = ResidualNet(
                n_loop, n_layer, 2, r_channels, 2 * r_channels, s_channels)
            self.proj1 = L.Convolution1D(
                s_channels, s_channels, 1, nobias=True)
            self.proj2 = L.Convolution1D(
                s_channels, a_channels, 1, nobias=True)
        self.a_channels = a_channels
        self.s_channels = s_channels
        self.use_embed_tanh = use_embed_tanh

    def forward(self, x, condition, generating=False):
        length = x.shape[2]
        x = self.embed(x)
        x = x[:, :, :length]  # crop
        if self.use_embed_tanh:
            x = F.tanh(x)
        z = F.relu(self.resnet(x, condition))
        z = F.relu(self.proj1(z))
        y = self.proj2(z)
        return y

    def initialize(self, n):
        self.resnet.initialize(n)

        self.embed.pad = 0
        self.embed_queue = chainer.Variable(self.xp.zeros(
            (n, self.a_channels, 2), dtype=self.embed.W.dtype))

        self.proj1_queue = chainer.Variable(self.xp.zeros(
            (n, self.s_channels, 1), dtype=self.proj1.W.dtype))

        self.proj2_queue3 = chainer.Variable(self.xp.zeros(
            (n, self.s_channels, 1), dtype=self.proj2.W.dtype))

    def generate(self, x, condition):
        self.embed_queue = F.concat((self.embed_queue[:, :, 1:], x), axis=2)
        x = self.embed(self.embed_queue)
        if self.use_embed_tanh:
            x = F.tanh(x)
        x = F.relu(self.resnet.generate(x, condition))

        self.proj1_queue = F.concat((self.proj1_queue[:, :, 1:], x), axis=2)
        x = F.relu(self.proj1(self.proj1_queue))

        self.proj2_queue3 = F.concat((self.proj2_queue3[:, :, 1:], x), axis=2)
        x = self.proj2(self.proj2_queue3)
        return x


class EncoderDecoderModel(chainer.Chain):
    def __init__(self, encoder, decoder):
        super(EncoderDecoderModel, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder

    def forward(self, x, condition):
        encoded_condition = self.encoder(condition)
        y = self.decoder(x, encoded_condition)
        return y
