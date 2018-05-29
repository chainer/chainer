import chainer
import numpy as np
import model


n_in = 784
n_units = 100
n_out = 10

# Create model object first
model1 = model.MLP(n_in, n_units, n_out)

chainer.serializers.load_npz('model.npz', model1)

print('model.npz loaded!')

model2 = model.MLP(n_in, n_units, n_out)

chainer.serializers.load_hdf5('model.h5', model2)

print('model.h5 loaded!')

model2_params = {name: param for name, param in model2.namedparams()}
for name, npz_param in model1.namedparams():
    h5_param = model2_params[name]
    np.testing.assert_array_equal(npz_param.array, h5_param.array)
    print(name, npz_param.shape)
