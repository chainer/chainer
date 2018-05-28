import chainer
import chainer.functions as F
import chainer.links as L
import h5py
import numpy as np

import model

n_in = 784
n_units = 100
n_out = 10

model = model.MLP(n_in, n_units, n_out)

# Save the model as a NPZ file
chainer.serializers.save_npz('model.npz', model)

print('model.npz saved!\n')

print('--- The list of saved params in model.npz ---')
saved_params = np.load('model.npz')
for param_key, param in saved_params.items():
    print(param_key, '\t:', param.shape)
print('---------------------------------------------\n')

# Save the model as a HDF5 archive
chainer.serializers.save_hdf5('model.h5', model)

print('model.h5 saved!')

print('--- The list of saved params in model.h5 ---')
f = h5py.File('model.h5', 'r')
for param_key, param in f.items():
    print('{}:'.format(param_key), end='')
    if isinstance(param, h5py.Dataset):
        print(' {}'.format(param.shape))
    else:
        print('')
    if isinstance(param, h5py.Group):
        for child_key, child in param.items():
            print('  {}:{}'.format(child_key, child.shape))
print('---------------------------------------------\n')
