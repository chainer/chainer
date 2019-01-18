Tips and FAQs
=============

GPU memory consumption is too high when used with CuPy
------------------------------------------------------

Both ChainerX and CuPy use their own GPU memory pools, meaning that GPU memory is not efficiently utilized (unused memory is kept without being freed by both ChainerX and CuPy).
You can run your script after setting the environment variable ``CHAINERX_CUDA_CUPY_SHARE_ALLOCATOR`` to ``1`` to use the experimental feature which makes sure that both ChainerX and CuPy share the same memory pool, hence reducing your peak GPU memory-usage.
You may also invoke ``chainerx._cuda.cupy_share_allocator`` instead of setting the environment variable for the same effect.
In this case, it is recommended to call the function prior to any GPU memory allocation.
