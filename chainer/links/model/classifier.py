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
        lossfun (callable):
            Loss function.
            You can specify one of loss functions from
            :doc:`built-in loss functions </reference/functions>`, or
            your own loss function (see the example below).
            It should not be an
            :doc:`loss functions with parameters </reference/links>`
            (i.e., :class:`~chainer.Link` instance).
            The function must accept two argument (an output from predictor
            and its ground truth labels), and return a loss.
            Returned value must be a Variable derived from the input Variable
            to perform backpropagation on the variable.
        accfun (callable):
            Function that computes accuracy.
            You can specify one of evaluation functions from
            :doc:`built-in evaluation functions </reference/functions>`, or
            your own evaluation function.
            The signature of the function is the same as ``lossfun``.
        label_key (int or str): Key to specify label variable from arguments.
            When it is ``int``, a variable in positional arguments is used.
            And when it is ``str``, a variable in keyword arguments is used.

    Attributes:
        predictor (~chainer.Link): Predictor network.
        lossfun (callable):
            Loss function.
            See the description in the arguments for details.
        accfun (callable):
            Function that computes accuracy.
            See the description in the arguments for details.
        y (~chainer.Variable): Prediction for the last minibatch.
        loss (~chainer.Variable): Loss value for the last minibatch.
        accuracy (~chainer.Variable): Accuracy for the last minibatch.
        compute_accuracy (bool): If ``True``, compute accuracy on the forward
            computation. The default value is ``True``.

    .. note::
        This link uses :func:`chainer.softmax_cross_entropy` with
        default arguments as a loss function (specified by ``lossfun``),
        if users do not explicitly change it. In particular, the loss function
        does not support double backpropagation.
        If you need second or higher order differentiation, you need to turn
        it on with ``enable_double_backprop=True``:

          >>> import chainer.functions as F
          >>> import chainer.links as L
          >>>
          >>> def lossfun(x, t):
          ...     return F.softmax_cross_entropy(
          ...         x, t, enable_double_backprop=True)
          >>>
          >>> predictor = L.Linear(10)
          >>> model = L.Classifier(predictor, lossfun=lossfun)

    """

    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(Classifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

    def forward(self, *args, **kwargs):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.
            kwargs (dict of ~chainer.Variable): Input minibatch.

        When ``label_key`` is ``int``, the corresponding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground truth
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        .. note::
            We set ``None`` to the attributes ``y``, ``loss`` and ``accuracy``
            each time before running the predictor, to avoid unnecessary memory
            consumption. Note that the variables set on those attributes hold
            the whole computation graph when they are computed. The graph
            stores interim values on memory required for back-propagation.
            We need to clear the attributes to free those values.

        Returns:
            ~chainer.Variable: Loss value.

        """

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.accuracy = None

        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
