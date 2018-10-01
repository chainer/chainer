import chainer
import h5py
import numpy as np

import model

# Create a model object first
model = model.MLP()


def save_parameters_as_npz(model, filename='model.npz'):
    # Save the model parameters into a NPZ file
    chainer.serializers.save_npz(filename, model)
    print('{} saved!\n'.format(filename))

    # Load the saved npz from NumPy and show the parameter shapes
    print('--- The list of saved params in model.npz ---')
    saved_params = np.load('model.npz')
    for param_key, param in saved_params.items():
        print(param_key, '\t:', param.shape)
    print('---------------------------------------------\n')


def save_parameters_as_hdf5(model, filename='model.h5'):
    # Save the model parameters into a HDF5 archive
    chainer.serializers.save_hdf5(filename, model)
    print('model.h5 saved!\n')

    # Load the saved HDF5 using h5py
    print('--- The list of saved params in model.h5 ---')
    f = h5py.File('model.h5', 'r')
    for param_key, param in f.items():
        msg = '{}:'.format(param_key)
        if isinstance(param, h5py.Dataset):
            msg += ' {}'.format(param.shape)
        print(msg)
        if isinstance(param, h5py.Group):
            for child_key, child in param.items():
                print('  {}:{}'.format(child_key, child.shape))
    print('---------------------------------------------\n')


save_parameters_as_npz(model)
save_parameters_as_hdf5(model)
