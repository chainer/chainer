import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L


def _augmentation(x):
    xp = cuda.get_array_module(x)
    MAX_SHIFT = 2
    batchsize, ch, h, w = x.shape
    h_shift, w_shift = xp.random.randint(-MAX_SHIFT, MAX_SHIFT + 1, size=2)
    a_h_sl = slice(max(0, h_shift), h_shift + h)
    a_w_sl = slice(max(0, w_shift), w_shift + w)
    x_h_sl = slice(max(0, - h_shift), - h_shift + h)
    x_w_sl = slice(max(0, - w_shift), - w_shift + w)
    a = xp.zeros(x.shape)
    a[:, :, a_h_sl, a_w_sl] = x[:, :, x_h_sl, x_w_sl]
    return a.astype(x.dtype)


def _count_params(m, n_grids=6):
    print('# of params', sum(param.size for param in m.params()))
    # The number of parameters in the paper (11.36M) might be
    # of the model with unshared matrices over primary capsules in a same grid
    # when input data are 36x36 images of MultiMNIST (n_grids = 10).
    # Our model with n_grids=10 has 11.349008M parameters.
    # (In the Sec. 4, the paper says "each capsule in the [6, 6] grid
    # is sharing their weights with each other.")
    print('# of params if unshared',
          sum(param.size for param in m.params()) +
          sum(param.size for param in m.Ws.params()) *
          (n_grids * n_grids - 1))


def squash(ss):
    ss_norm2 = F.sum(ss ** 2, axis=1, keepdims=True)
    """
    # ss_norm2 = F.broadcast_to(ss_norm2, ss.shape)
    # vs = ss_norm2 / (1. + ss_norm2) * ss / F.sqrt(ss_norm2): naive
    """
    norm_div_1pnorm2 = F.sqrt(ss_norm2) / (1. + ss_norm2)
    norm_div_1pnorm2 = F.broadcast_to(norm_div_1pnorm2, ss.shape)
    vs = norm_div_1pnorm2 * ss  # :efficient
    # (batchsize, 16, 10)
    return vs


def get_norm(vs):
    return F.sqrt(F.sum(vs ** 2, axis=1))


init = chainer.initializers.Uniform(scale=0.05)


class CapsNet(chainer.Chain):

    def __init__(self, use_reconstruction=False):
        super(CapsNet, self).__init__()
        self.n_iterations = 3  # dynamic routing
        self.n_grids = 6  # grid width of primary capsules layer
        self.n_raw_grids = self.n_grids
        self.use_reconstruction = use_reconstruction
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 256, ksize=9, stride=1,
                                         initialW=init)
            self.conv2 = L.Convolution2D(256, 32 * 8, ksize=9, stride=2,
                                         initialW=init)
            self.Ws = chainer.ChainList(
                *[L.Convolution2D(8, 16 * 10, ksize=1, stride=1, initialW=init)
                  for i in range(32)])

            self.fc1 = L.Linear(16 * 10, 512, initialW=init)
            self.fc2 = L.Linear(512, 1024, initialW=init)
            self.fc3 = L.Linear(1024, 784, initialW=init)

        _count_params(self, n_grids=self.n_grids)
        self.results = {'N': 0, 'loss': [], 'correct': [],
                        'cls_loss': [], 'rcn_loss': []}

    def pop_results(self):
        merge = dict()
        merge['mean_loss'] = sum(self.results['loss']) / self.results['N']
        merge['cls_loss'] = sum(self.results['cls_loss']) / self.results['N']
        merge['rcn_loss'] = sum(self.results['rcn_loss']) / self.results['N']
        merge['accuracy'] = sum(self.results['correct']) / self.results['N']
        self.results = {'N': 0, 'loss': [], 'correct': [],
                        'cls_loss': [], 'rcn_loss': []}
        return merge

    def __call__(self, x, t):
        if chainer.config.train:
            x = _augmentation(x)
        vs_norm, vs = self.output(x)
        self.loss = self.calculate_loss(vs_norm, t, vs, x)

        self.results['loss'].append(self.loss.data * t.shape[0])
        self.results['correct'].append(self.calculate_correct(vs_norm, t))
        self.results['N'] += t.shape[0]
        return self.loss

    def output(self, x):
        batchsize = x.shape[0]
        n_iters = self.n_iterations
        gg = self.n_grids * self.n_grids

        # h1 = F.relu(self.conv1(x))
        h1 = F.leaky_relu(self.conv1(x), 0.05)
        pr_caps = F.split_axis(self.conv2(h1), 32, axis=1)
        # shapes if MNIST. -> if MultiMNIST
        # x (batchsize, 1, 28, 28) -> (:, :, 36, 36)
        # h1 (batchsize, 256, 20, 20) -> (:, :, 28, 28)
        # pr_cap (batchsize, 8, 6, 6) -> (:, :, 10, 10)

        Preds = []
        for i in range(32):
            pred = self.Ws[i](pr_caps[i])
            Pred = pred.reshape((batchsize, 16, 10, gg))
            Preds.append(Pred)
        Preds = F.stack(Preds, axis=3)
        assert(Preds.shape == (batchsize, 16, 10, 32, gg))

        bs = self.xp.zeros((batchsize, 10, 32, gg), dtype='f')
        for i_iter in range(n_iters):
            cs = F.softmax(bs, axis=1)
            Cs = F.broadcast_to(cs[:, None], Preds.shape)
            assert(Cs.shape == (batchsize, 16, 10, 32, gg))
            ss = F.sum(Cs * Preds, axis=(3, 4))
            vs = squash(ss)
            assert(vs.shape == (batchsize, 16, 10))

            if i_iter != n_iters - 1:
                Vs = F.broadcast_to(vs[:, :, :, None, None], Preds.shape)
                assert(Vs.shape == (batchsize, 16, 10, 32, gg))
                bs = bs + F.sum(Vs * Preds, axis=1)
                assert(bs.shape == (batchsize, 10, 32, gg))

        vs_norm = get_norm(vs)
        return vs_norm, vs

    def reconstruct(self, vs, t):
        xp = self.xp
        batchsize = t.shape[0]
        I = xp.arange(batchsize)
        mask = xp.zeros(vs.shape, dtype='f')
        mask[I, :, t] = 1.
        masked_vs = mask * vs

        x_recon = F.sigmoid(
            self.fc3(F.relu(
                self.fc2(F.relu(
                    self.fc1(masked_vs)))))).reshape((batchsize, 1, 28, 28))
        return x_recon

    def calculate_loss(self, vs_norm, t, vs, x):
        class_loss = self.calculate_classification_loss(vs_norm, t)
        self.results['cls_loss'].append(class_loss.data * t.shape[0])
        if self.use_reconstruction:
            recon_loss = self.calculate_reconstruction_loss(vs, t, x)
            self.results['rcn_loss'].append(recon_loss.data * t.shape[0])
            return class_loss + 0.0005 * recon_loss
        else:
            return class_loss

    def calculate_classification_loss(self, vs_norm, t):
        xp = self.xp
        batchsize = t.shape[0]
        I = xp.arange(batchsize)
        T = xp.zeros(vs_norm.shape, dtype='f')
        T[I, t] = 1.
        m = xp.full(vs_norm.shape, 0.1, dtype='f')
        m[I, t] = 0.9

        loss = T * F.relu(m - vs_norm) ** 2 + \
            0.5 * (1. - T) * F.relu(vs_norm - m) ** 2
        return F.sum(loss) / batchsize

    def calculate_reconstruction_loss(self, vs, t, x):
        batchsize = t.shape[0]
        x_recon = self.reconstruct(vs, t)
        loss = (x_recon - x) ** 2
        return F.sum(loss) / batchsize

    def calculate_correct(self, v, t):
        return (self.xp.argmax(v.data, axis=1) == t).sum()
