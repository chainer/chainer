Comparison with Other Frameworks
================================

A table for quick comparison
----------------------------

This table compares Chainer with other actively developed deep learning frameworks. Content is current as of July 2017.

.. csv-table::
   :stub-columns: 2
   :header: ,,"`Chainer <https://github.com/chainer/chainer>`_","`PyTorch <https://github.com/pytorch/pytorch>`_","`TensorFlow <https://github.com/tensorflow/tensorflow>`_","`Theano <https://github.com/Theano/Theano>`_-based","`Caffe1 <https://github.com/bvlc/caffe>`_/`Caffe2 <https://github.com/caffe2/caffe2>`_","`Torch7 <https://github.com/torch/torch>`_","`MXNet <https://github.com/dmlc/mxnet>`_","`DyNet <https://github.com/clab/dynet>`_","`PaddlePaddle <https://github.com/PaddlePaddle/Paddle>`_","`DL4J <https://github.com/deeplearning4j/deeplearning4j>`_","`CNTK <https://github.com/Microsoft/cntk>`_","`neon <https://github.com/NervanaSystems/neon>`_","`Knet.jl <https://github.com/denizyuret/Knet.jl>`_","`Darknet <https://github.com/pjreddie/darknet>`_","`Thinc <https://github.com/explosion/thinc>`_"
   
   "Basics","Language","Python","Python","Python","Python","Python/C++/ MATLAB","LuaJIT","Python/others","Python/C++","Python/C++","Java","BrainScript/ Python/C++","Python","Julia","C","Python"
   ,"Approach","define-by-run","define-by-run","symbolic autograd","symbolic autograd","static","static/ manual grads","symbolic autograd/ manual grads/ define-by-run [1]_","define-by-run","symbolic autograd","static/ manual grads/ symbolic autograd [2]_","static/ symbolic autograd","static/ symbolic autograd [3]_","define-by-run","static","callback-based define-by-run"
   ,"CPU backend package","NumPy","`TH <https://github.com/torch/torch>`_","`Eigen <https://github.com/PX4/eigen>`_","NumPy",,"TH","`mshadow <https://github.com/dmlc/mshadow>`_","Eigen",,"`ND4J <https://github.com/deeplearning4j/nd4j>`_",,"NumPy","`Julia <https://github.com/julialang/julia>`_",,"NumPy"
   ,"GPU backend package","`CuPy <https://github.com/cupy/cupy>`_","`THC <https://github.com/torch/cutorch>`_","Eigen","`libgpuarray <https://github.com/Theano/libgpuarray>`_",,"THC","mshadow","Eigen",,"ND4J",,"neon",KnetArrays,,"CuPy"
   ,"Primary sponsor","Preferred Networks","Facebook","Google","MILA","Facebook","Facebook","Amazon/Apache","CMU","Baidu","Skymind","Microsoft","Intel Nervana","Koç University","Joe Redmon","Explosion AI"
   "NNs","CNNs","full","full","full","full","full","full","full","partial","full","full","full","full","partial","full","none"
   ,"RNNs","full","full","full","full","partial","full","full","full","full","full","full","partial","partial","partial","partial"
   ,"Reverse-mode autograd","Y","Y","Y","Y",,"`torch-autograd <https://github.com/twitter/torch-autograd>`_","Y","Y","Y",,"Y","`ngraph <https://github.com/NervanaSystems/ngraph>`_","Y",,"with closures"
   ,"Forward-mode autograd",,,"`tensorflow-forward-ad <https://github.com/renmengye/tensorflow-forward-ad>`_","Y",,,,,,,,,,,
   ,"Higher-order grads",,"Y","Y","Y",,,,,,,,,"Y",,
   ,"Variable-length loops","native","native","while_loop","scan","RNNs only","native","2017","native","RNNs only","none","dynamic axis","none","native","none","native"
   ,"Different architectures per batch","native","native","`fold <https://github.com/tensorflow/fold>`_",,,"torch-autograd","`MinPy <https://github.com/dmlc/MinPy>`_","native",,,,,"native",,"native"
   "Performance","cuDNN support","full","full","partial","partial","full","full","full","partial","full","partial","full","N/A [4]_",,"partial",
   ,"CPU/GPU generic backend","Y","Y",,,,"Y","Y","Y","Y","Y","Y","Y","Y",,"Y"
   ,"Multi-GPU data parallelism","Y","Y","Y","Y","Y","Y","Y",,"Y","Y","Y","Y","Y","Y",
   ,"Multi-GPU model parallelism","Y","Y","Y","Y","Y","Y","Y",,"Y",,"Y","Y",,,
   ,"Multiprocessing [5]_","full","partial",,,,,,"full",,,,,,,
   ,"Distributed training","`ChainerMN <https://github.com/chainer/chainermn>`_","THD","Y",,2017,"`torch-distlearn <https://github.com/twitter/torch-distlearn>`_","Y",,"Y","Spark","Y","Y",,,
   "Misc","Runtime debugging","debug mode, typechecking, pdb","pdb","tfdbg",,,,"Monitor","pdb",,"Java debuggers","cntk.debugging",,"Gallium.jl","gdb","pdb"
   ,"Trainer abstraction","native","`tnt <https://github.com/pytorch/tnt>`_",,"`Blocks <https://github.com/mila-udem/blocks>`_, `Lasagne <https://github.com/Lasagne/Lasagne>`_, `Keras <https://github.com/fchollet/keras>`_","native","`torchnet <https://github.com/torchnet/torchnet>`_",,,"native","native","native","native",,,"native"
   ,"Reporter abstraction","native","tnt","native",,,"torchnet","native",,,"native","native",,,,
   ,"Web interface",,,"`TensorBoard <https://github.com/tensorflow/tensorboard>`_",,,,,,,"DL4J-UI",,"Nervana Cloud",,,
   ,"Graph compilation engine",,2017,"`XLA <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/>`_",,2017,,"`NNVM <https://github.com/dmlc/nnvm>`_",,,,,"ngraph",,,

.. [1] Define-by-run is in development as of June 2017 and tracked in `dmlc/mxnet#5705 <https://github.com/dmlc/mxnet/pull/5705>`_. It is also possible using the much slower MinPy extension.
.. [2] Symbolic autograd is in development as of June 2017 and tracked in `deeplearning4j/nd4j#1750 <https://github.com/deeplearning4j/nd4j/pull/1750>`_.
.. [3] Symbolic autograd is available only with ngraph backend (experimental).
.. [4] Nervana provides kernels that are meant to compete with cuDNN.
.. [5] Multiprocessing provides a significant performance improvement only for frameworks that use Python at runtime.

Benchmarks
----------

Benchmarks for convolutional networks can be found at `convnet-benchmarks <https://github.com/soumith/convnet-benchmarks>`_ while some NLP benchmarks are at `dynet-benchmark <https://github.com/neulab/dynet-benchmark>`_. Chainer wraps the latest available cuDNN kernels for CNNs and RNNs, so performance of most common networks that use these kernels is typically similar to that of other modern frameworks. As Chainer's define-by-run approach means the user's Python code is executed directly at runtime, particularly complex networks or those with very small tensor sizes may be slower than in static-graph frameworks.
