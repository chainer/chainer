from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class Classifier(link.Chain):

    """A simple classifier model.

    This is an example of chain that wraps another chain. It computes the
    loss and accuracy based on a given input/label pair.

    Args:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        label_key (int or str): Key to specify label variable from arguments.
            When it is ``int``, a variable in positional arguments is used.
            And when it is ``str``, a variable in keyword arguments is used.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (function): Loss function.
        accfun (function): Function that computes accuracy.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.

    """

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy):
        super(Classifier, self).__init__(predictor=predictor)
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None

    def __call__(self, *args):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.
            kwargs (dict of ~chainer.Variable): Input minibatch.

        When ``label_key`` is ``int``, the correpoding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground trush
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """

        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
