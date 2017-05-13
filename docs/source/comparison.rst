Comparison with Other Frameworks
================================

A table for quick comparison
----------------------------

This table compares Chainer with other actively developed deep learning frameworks. Content is current as of May 2017.

.. csv-table::
   :stub-columns: 2
   :header: ,,"Chainer","PyTorch","TensorFlow","Theano-based","Caffe1/2","Torch7","MXNet","DyNet","PaddlePaddle","DL4J","CNTK","neon","Knet.jl","Darknet","Thinc"
   
   "Basics","Language","Python","Python","Python","Python","Python/C++","LuaJIT","Python/others","Python/C++","Python/C++","Java","BrainScript/ Python/C++","Python","Julia","C","Python"
   ,"Approach","define-by-run","define-by-run","symbolic autograd","symbolic autograd","static","static/ manual grads","symbolic autograd/ manual grads/ define-by-run [1]_","define-by-run","symbolic autograd","static/ manual grads","static/ symbolic autograd","static/ symbolic autograd [2]_","define-by-run","static","callback-based define-by-run"
   ,"CPU backend","NumPy","TH","Eigen","NumPy","custom","TH","mshadow","Eigen","custom","ND4J","custom","neon","Julia","custom","NumPy"
   ,"GPU backend","CuPy","THC","Eigen","libgpuarray","custom","THC","mshadow","Eigen","custom","ND4J","custom","neon","custom","custom","CuPy"
   ,"Primary sponsor","Preferred Networks","Facebook","Google","MILA","Facebook","Facebook","Amazon/Apache","CMU","Baidu","Skymind","Microsoft","Intel Nervana","Koç University","Joe Redmon","Explosion AI"
   "NNs","CNNs","full","full","full","full","full","full","full","partial","partial","full","full","full","partial","full","none"
   ,"RNNs","full","full","full","full","partial","full","full","full","full","partial","full","partial","partial","partial","partial"
   ,"Reverse-mode autograd","Y","Y","Y","Y",,"torch-autograd","Y","Y","Y",,"Y","ngraph","Y",,"with closures"
   ,"Forward-mode autograd",,,,"Y",,,,,,,,,,,
   ,"Higher-order grads",,"Y","Y","Y",,,,,,,,,"Y",,
   ,"Variable-length loops","native","native","while_loop","scan","RNNs only","native","2017","native","RNNs only","none","dynamic axis","none","native","none","native"
   ,"Per-batch architectures","native","native","fold",,,"torch-autograd","MinPy","native",,,,,"native",,"native"
   "Performance","cuDNN support","full","full","partial","partial","full","full","full","partial","full","partial","full","N/A [3]_",,"partial",
   ,"CPU/GPU generic backend","Y","Y",,,,"Y","Y","Y","Y","Y","Y","Y","Y","Y","Y"
   ,"Multi-GPU data parallelism","Y","Y","Y","Y","Y","Y","Y",,"Y","Y","Y","Y",,,
   ,"Multi-GPU model parallelism","Y","Y","Y","Y","Y","Y","Y",,"Y",,"Y","Y",,,
   ,"Multiprocessing [4]_","[#2213]","partial",,,,,,"full",,,,,,,
   ,"Distributed training","2Q 2017","2Q 2017","Y",,2017,"torch-distlearn","Y",,"Y","Y","Y","Y",,,
   "Misc","Runtime debugging","debug mode, typechecking, pdb","pdb","tfdbg",,,,"Monitor","pdb",,,"cntk.debugging",,"Gallium.jl","gdb","pdb"
   ,"Trainer abstraction","native","tnt",,"various packages","native","torchnet",,,"native","native","native","native",,,"native"
   ,"Reporter abstraction","native","tnt","native",,,"torchnet","native",,,"native","native",,,,
   ,"Web interface",,,"TensorBoard",,,,,,,"DL4J-UI",,"Nervana Cloud",,,
   ,"Graph compilation engine",,2017,"XLA",,2017,,"NNVM",,,,,"ngraph",,,

.. [1] Define-by-run is in development as of May 2017 and tracked in `this pull request <https://github.com/dmlc/mxnet/pull/5705>`_. It is also possible using the much slower MinPy extension.
.. [2] Symbolic autograd is available only with ngraph backend (in development).
.. [3] Nervana provides kernels that are meant to compete with cuDNN.
.. [4] Multiprocessing provides a significant performance improvement only for frameworks that use Python at runtime.

Benchmarks
----------

Benchmarks for convolutional networks can be found at `convnet-benchmarks <https://github.com/soumith/convnet-benchmarks>`_ while some NLP benchmarks are at `dynet-benchmark <https://github.com/neulab/dynet-benchmark>`_. Chainer wraps the latest available cuDNN kernels for CNNs and RNNs, so performance of most common networks that use these kernels is typically similar to that of other modern frameworks. As Chainer's define-by-run approach means the user's Python code is executed directly at runtime, particularly complex networks or those with very small tensor sizes may be slower than in static-graph frameworks.
