from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter

from chainer.link_hooks.soft_nearest_neighbor_loss import SNNL_hook


class ClassifierSNNL(link.Chain):
    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 factor=-10,
                 link_names=None
                 ):
        super(ClassifierSNNL, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.factor = factor

        self.snnl_hooks = []

        if link_names is None:
            for l in list(predictor.children())[:-1]:
                hook = SNNL_hook()
                hook.name = '{}_snn_loss'.format(l.name)
                l.add_hook(hook)
                self.snnl_hooks.append(hook)
        else:
            for l in predictor.children():
                if l.name in link_names:
                    hook = SNNL_hook()
                    hook.name = '{}_snn_loss'.format(l.name)
                    l.add_hook(hook)
                    self.snnl_hooks.append(hook)

        with self.init_scope():
            self.predictor = predictor

    def forward(self, x, t):
        for snnl_hook in self.snnl_hooks:
            snnl_hook.set_t(t)

        self.y = self.predictor(x)
        self.loss = self.lossfun(self.y, t)
        self.snn_loss = self.loss
        self.accuracy = self.accfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        reporter.report({'accuracy': self.accuracy}, self)
        for snnl_hook in self.snnl_hooks:
            self.snn_loss += self.factor * snnl_hook.get_loss()
            reporter.report({snnl_hook.name: snnl_hook.get_loss()}, self)
        reporter.report({'snn_loss': self.snn_loss}, self)
        return self.snn_loss
