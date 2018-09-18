import numpy as np

import chainer
from chainer import cuda
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
from chainer import reporter


class AvgELBOLoss(chainer.Chain):
    def __init__(self, encoder, decoder, prior, beta=1.0):
        super(AvgELBOLoss, self).__init__()
        self.beta = beta

        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
            self.prior = prior

        self.loss = None
        self.rec = None
        self.penalty = None

    def __call__(self, x):
        q_z = self.encoder(x)
        z = q_z.sample()
        p_x = self.decoder(z)
        p_z = self.prior()

        self.loss = None
        self.rec = None
        self.penalty = None
        self.rec = F.mean(F.sum(p_x.log_prob(x), axis=-1))
        self.penalty = F.mean(
            F.sum(q_z.log_prob(z) - p_z.log_prob(z), axis=-1))
        self.loss = - (self.rec - self.beta * self.penalty)
        reporter.report({'loss': self.loss}, self)
        reporter.report({'rec': self.rec}, self)
        reporter.report({'penalty': self.penalty}, self)
        return self.loss


class Encoder(chainer.Chain):

    def __init__(self, n_in, n_latent, n_h):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(n_in, n_h)
            self.mu = L.Linear(n_h, n_latent)
            self.ln_sigma = L.Linear(n_h, n_latent)

    def forward(self, x):
        h = F.tanh(self.linear(x))
        mu = self.mu(h)
        ln_sigma = self.ln_sigma(h)  # log(sigma)
        return D.Normal(loc=mu, log_scale=ln_sigma)


class Decoder(chainer.Chain):

    def __init__(self, n_in, n_latent, n_h):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(n_latent, n_h)
            self.output = L.Linear(n_h, n_in)

    def forward(self, z):
        h = F.tanh(self.linear(z))
        h = self.output(h)
        return D.Bernoulli(logit=h)


class Prior(chainer.Chain):

    def __init__(self, n_latent, dtype=np.float32, device=-1):
        super(Prior, self).__init__()

        loc = np.zeros(n_latent, dtype=dtype)
        scale = np.ones(n_latent, dtype=dtype)
        if device != -1:
            loc = cuda.to_gpu(loc, device=device)
            scale = cuda.to_gpu(scale, device=device)

        self.loc = chainer.Variable(loc)
        self.scale = chainer.Variable(scale)

    def forward(self):
        return D.Normal(self.loc, scale=self.scale)


def make_encoder(n_in, n_latent, n_h):
    return Encoder(n_in, n_latent, n_h)


def make_decoder(n_in, n_latent, n_h):
    return Decoder(n_in, n_latent, n_h)


def make_prior(n_latent, dtype=np.float32, device=-1):
    return Prior(n_latent, dtype=dtype, device=device)
