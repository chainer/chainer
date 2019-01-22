Serializers -- saving and loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Serializer is a simple interface to serialize or deserialize an object.
:class:`~chainer.Link`, :class:`~chainer.Optimizer`, and :class:`~chainer.training.Trainer` support serialization.

Concrete serializers are defined in the :mod:`~chainer.serializers` module.
It supports NumPy NPZ and HDF5 formats.

For example, we can serialize a link object into NPZ file by the :func:`~chainer.serializers.save_npz` function:

Assuming we have defined a ``model``:

.. code-block:: python

   >>> from chainer import serializers
   >>> serializers.save_npz('my.model', model)

This saves the parameters of ``model`` into the file ``'my.model'`` in NPZ format.
The saved model can be read back from ``my.model`` back into ``model``  by the :func:`~chainer.serializers.load_npz` function:

.. code-block:: python

   >>> serializers.load_npz('my.model', model)

.. note::
   Note that only the parameters and the *persistent values* are serialized by this serialization code.
   Other attributes are not saved automatically.
   You can register arrays, scalars, or any serializable objects as persistent values by the :meth:`~chainer.Link.add_persistent` method.
   The registered values can be accessed by attributes of the name passed to the add_persistent method.

The state of an optimizer can also be saved by the same functions:

.. code-block:: python

   >>> serializers.save_npz('my.state', optimizer)
   >>> serializers.load_npz('my.state', optimizer)

.. note::
   Note that serialization of optimizer only saves its internal states including number of iterations, momentum vectors of MomentumSGD, etc.
   It does not save the parameters and persistent values of the target link.
   We have to explicitly save the target link with the optimizer to resume the optimization from saved states.
   This can be done by saving the entire :class:`~training.Trainer` object, like this:

.. code-block:: python

   >>> serializers.save_npz('my.state', trainer)

Support of the HDF5 format is enabled if the h5py package is installed.
Serialization and deserialization with the HDF5 format are almost identical to those with the NPZ format;
just replace :func:`~chainer.serializers.save_npz` and :func:`~chainer.serializers.load_npz` by :func:`~chainer.serializers.save_hdf5` and :func:`~chainer.serializers.load_hdf5`, respectively.
