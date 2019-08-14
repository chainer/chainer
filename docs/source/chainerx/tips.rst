Tips and FAQs
=============

Can I use ChainerX without Chainer?
-----------------------------------

Yes, it is possible. See the code samples below.

- Train an MLP with MNIST dataset (:tree:`chainerx_cc/examples/mnist_py`)
- Train a CNN with ImageNet dataset (:tree:`chainerx_cc/examples/imagenet_py`)

What does the C++ interface look like?
--------------------------------------

It is almost identical to the Python interface with a 1-to-1 mapping.
The interface is still subject to change, but there is an example code:

- Train an MLP with MNIST dataset in C++ (:tree:`chainerx_cc/examples/mnist`)

GPU memory consumption is too high when used with CuPy
------------------------------------------------------

Both ChainerX and CuPy use their own GPU memory pools, meaning that GPU memory is not efficiently utilized (unused memory is kept without being freed by both ChainerX and CuPy).
You can run your script after setting the environment variable ``CHAINERX_CUDA_CUPY_SHARE_ALLOCATOR`` to ``1`` to use the experimental feature which makes sure that both ChainerX and CuPy share the same memory pool, hence reducing your peak GPU memory-usage.
You may also invoke ``chainerx._cuda.cupy_share_allocator`` instead of setting the environment variable for the same effect.
In this case, it is recommended that you call the function prior to any GPU memory allocation.
