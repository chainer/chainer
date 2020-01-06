.. _module:

Module Reference
================

.. module:: onnx_chainer

Export
------

ONNX-Chainer exports Chainer model to ONNX graph with various options.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   onnx_chainer.export
   onnx_chainer.export_testcase


Export Utilities
----------------

ONNX-Chainer provides some utility functions to help exporting.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   onnx_chainer.replace_func.fake_as_funcnode
   onnx_chainer.replace_func.as_funcnode


Convert Utilities
-----------------

These utilities helps converting from Chainer model to ONNX format, mainly used them internally.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   onnx_chainer.context.Context


.. autosummary::
   :toctree: generated/
   :nosignatures:

   onnx_chainer.onnx_helper.GraphBuilder
   onnx_chainer.onnx_helper.set_func_name
   onnx_chainer.onnx_helper.get_func_name
   onnx_chainer.onnx_helper.make_node
   onnx_chainer.onnx_helper.write_tensor_pb
   onnx_chainer.onnx_helper.cleanse_param_name


Testing Utilities
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   onnx_chainer.testing.input_generator.increasing
   onnx_chainer.testing.input_generator.nonzero_increasing
   onnx_chainer.testing.input_generator.positive_increasing



