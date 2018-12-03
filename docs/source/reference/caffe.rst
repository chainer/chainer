Caffe Model Support
===================

.. module:: chainer.links.caffe

`Caffe <http://caffe.berkeleyvision.org/>`_ is a popular framework maintained by `BVLC <http://bvlc.eecs.berkeley.edu/>`_ at UC Berkeley.
It is widely used by computer vision communities, and aims at fast computation and easy usage without any programming.
The BVLC team provides trained reference models in their `Model Zoo <http://caffe.berkeleyvision.org/model_zoo.html>`_, which can reduce training time required for a new task.

Import
------

Chainer can import the reference models and emulate the network by :class:`~chainer.Link` implementations.
This functionality is provided by the :class:`chainer.links.caffe.CaffeFunction` class.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.links.caffe.CaffeFunction


Export
------

.. module:: chainer.exporters

Chainer can export a model from :class:`~chainer.Link`.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   chainer.exporters.caffe.export

