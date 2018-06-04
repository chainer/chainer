import chainer
import numpy as np

import model


def load_npz_file_to_model(npz_filename, model_object):
    chainer.serializers.load_npz(npz_filename, model_object)
    print('{} loaded!'.format(npz_filename))

# Create model object first
model1 = model.MLP()

# Load the saved parameters into the model object
load_npz_file_to_model('model.npz', model1)


def load_hdf5_file_to_model(hdf5_filename, model_object):
    chainer.serializers.load_hdf5(hdf5_filename, model_object)
    print('{} loaded!'.format(hdf5_filename))

# Create another model object first
model2 = model.MLP()

# Load the saved parameters into the model object
load_hdf5_file_to_model('model.h5', model2)

# Check that the loaded parameters are same
model2_params = {name: param for name, param in model2.namedparams()}
for name, npz_param in model1.namedparams():
    h5_param = model2_params[name]
    np.testing.assert_array_equal(npz_param.array, h5_param.array)
    print(name, npz_param.shape)

