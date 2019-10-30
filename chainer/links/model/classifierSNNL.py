from chainer import configuration
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
import chainer.link_hooks
from chainer import reporter


class ClassifierSNNL(link.Chain):

    """A simple classifier model with soft nearest neighbor loss.

    See: `Analyzing and Improving Representations
    with the Soft Nearest Neighbor Loss
    <https://arxiv.org/abs/1902.01889>`_.

    A combination loss of soft nearest neighbor loss calculated at every layer
    in the network, and standard cross entropy of the logits.


    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (callable): Loss function.
        accfun (callable): Function that computes accuracy.
        factor (float32):
            The balance factor between SNNL and ross Entropy. If factor is
            negative, then SNNL will be maximized.
        link_names(list):
            The names of the layers at which to calculate SNNL.
            If not provided, then SNNL is applied to each internal layer.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (callable): Loss function.
        accfun (callable): Function that computes accuracy.
        factor (float32): The balance factor between SNNL and ross Entropy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        snns_loss (~chainer.Variable):
            Combination loss of Soft Nearest Neighbor Loss calculated at every
            layer in the network, and standard cross entropy of the logits.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        snns_hooks: List of snns hook link.

    """

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

        with self.init_scope():
            self.predictor = predictor

        if configuration.config.train is False:
            return

        if link_names is None:
            for l in list(predictor.children())[:-1]:
                hook = chainer.link_hooks.SNNL_hook()
                hook.name = '{}_snn_loss'.format(l.name)
                l.add_hook(hook)
                self.snnl_hooks.append(hook)
        else:
            for l in predictor.children():
                if l.name in link_names:
                    hook = chainer.link_hooks.SNNL_hook()
                    hook.name = '{}_snn_loss'.format(l.name)
                    l.add_hook(hook)
                    self.snnl_hooks.append(hook)

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
