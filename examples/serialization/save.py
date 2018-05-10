import chainer
import chainer.functions as F
import chainer.links as L

n_units = 100
n_out = 10

# How to save Sequential model
model = chainer.Sequential(
    L.Linear(784, n_units),
    F.relu,
    L.Linear(n_units, n_units),
    F.relu,
    L.Linear(n_units, n_out),
)

# Save the model as a NPZ file
chainer.serializers.save_npz('mlp_sequential.npz', model)

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)
