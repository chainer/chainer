.. module:: chainer.serializers

Serializers
===========

Serialization in NumPy NPZ format
---------------------------------

NumPy serializers can be used in arbitrary environments that Chainer runs with.
It consists of asymmetric serializer/deserializer due to the fact that :func:`numpy.savez` does not support online serialization.
Therefore, serialization requires two-step manipulation: first packing the objects into a flat dictionary, and then serializing it into npz format.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.serializers.DictionarySerializer
   chainer.serializers.NpzDeserializer
   chainer.serializers.save_npz
   chainer.serializers.load_npz

Serialization in HDF5 format
----------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.serializers.HDF5Serializer
   chainer.serializers.HDF5Deserializer
   chainer.serializers.save_hdf5
   chainer.serializers.load_hdf5
