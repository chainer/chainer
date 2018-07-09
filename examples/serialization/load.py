import chainer
import numpy as np

import model


def load_npz_file_to_model(npz_filename='model.npz'):
    # Create model object first
    model1 = model.MLP()

    # Load the saved parameters into the model object
    chainer.serializers.load_npz(npz_filename, model1)
    print('{} loaded!'.format(npz_filename))

    return model1


def load_hdf5_file_to_model(hdf5_filename='model.h5'):
    # Create another model object first
    model2 = model.MLP()

    # Load the saved parameters into the model object
    chainer.serializers.load_hdf5(hdf5_filename, model2)
    print('{} loaded!'.format(hdf5_filename))

    return model2


model1 = load_npz_file_to_model()
model2 = load_hdf5_file_to_model()

# Check that the loaded parameters are same
model2_params = {name: param for name, param in model2.namedparams()}
for name, npz_param in model1.namedparams():
    h5_param = model2_params[name]
    np.testing.assert_array_equal(npz_param.array, h5_param.array)
    print(name, npz_param.shape)
