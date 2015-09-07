import chainer
import chainer.functions as F


class MLP(chainer.ModelList):

    """Multilayer Perceptron model for MNIST dataset.

    This is a very simple example of ModelList. This implementation is a
    demonstration of simplicity of writing a model. It lacks flexibility to
    select number of layers and hidden units, and the type of activation
    function.

    """
    def __init__(self):
        super(MLP, self).__init__(
            F.Linear(784, 1000),  # the first layer
            F.Linear(1000, 1000),  # the second layer
            F.Linear(1000, 10),  # the last layer
        )

    def predict(self, x, train=True):
        h1 = F.dropout(F.relu(self[0](x)), train=train)
        h2 = F.dropout(F.relu(self[1](h1)), train=train)
        return self[2](h2)

    def evaluate(self, x, t, train=True):
        y = self.predict(x, train)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
