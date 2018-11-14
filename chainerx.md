# ChainerX User Guide

*ChainerX*, or `chainerx` is a standalone Python package that's integrated into Chainer.
It is implemented almost purely in C++ where the C++ implementations are exposed via Python bindings, allowing Chainer to make use of it.
It does in other words **not** replace Chainer. It aims to instead improve the performance of Chainer in terms of speed by reducing the Python overhead.

This guide is aimed toward users familiar with the Chainer interface but want to improve their training/inference speed, using ChainerX.
It explains how to install it, the motivation behind it and how to modify your existing Chainer code to use ChainerX.

- [Installation](#installation)
- [About ChainerX](#about-chainerx)
- [Making your script use ChainerX](#making-your-script-use-chainerx)
- [Known issues](#known-issues)
- [FAQ](#faq)

## Installation

See [ChainerX installation](docs/source/chainerx/install/index.rst).

## About ChainerX

Currently, ChainerX may be thought of as an alternative to NumPy and CuPy with automatic differentiation, i.e. graph construction and back-propagation built-in.
That is, a NumPy/CuPy array with `Variable` properties.

It can still be wrapped in a `Variable` and passed to any existing Chainer code.

```python
import chainer as ch
import chainerx as chx

arr = chx.array([1, 2, 3], dtype='f')
var = ch.Variable(arr)

# Your Chainer code...
```

Following `chainer.functions` functions operating on the `var` resulting in an extension of the graph will call the corresponding graph constructions APIs defined in the C++ layer, working around the Python function calls.
Similarly, calling `Variable.backward` on any resulting variable will delegate the work to C++ by calling `chainerx.ndarray.backward` spending little time in the Python world.

### NumPy/CuPy fallback

As the features above require ChainerX to support each `FunctionNode` implementation in Chainer, ChainerX utilizes a fallback mechanism while gradually extending the support, instead of e.g. raising an unsupported error.
This approach is taken because the integration with Chainer takes time and we **do not want existing Chainer users to have to make severe changes to their code bases in order to try ChainerX**.
The fallback logic simply casts the `chainerx.ndarray`s inside the `Variable` to `numpy.ndarray`s or `cupy.ndarray`s (without copy) and calls the forward and backward methods respectively.
For a complete list of supported ChainerX functions please refer to [this page](chainerx_cc/chainerx/python/routines.cc) as those in many cases have corresponding `chainer.functions` functions.
Similar fallback conversions are found throughout the code outside the `FunctionNode` as well during the integration.

### Backends and devices

Chainer distinguishes between CPU (a.k.a. native) and CUDA arrays using NumPy and CuPy.
ChainerX arrays on the other hand may be allocated on any device on any backend.
If you're working with `Variable.array` directly, you may have to be aware of the following interfaces.
Otherwise, they are handled by the `Variable` and similarly for other classes.

```python
arr = chx.array([1, 2, 3], dtype='f', device='native')  # Native backend
arr.device  # == native:0

arr = chx.array([1, 2, 3], dtype='f', device='cuda:1')  # CUDA backend, second device
arr.device  # == cuda:0
```

This allows third-party backends to be plugged into ChainerX. It is outside the scope of this document but can be achieved by implementing the [`chainerx::Backend`](chainerx_cc/chainerx/backend.h) and [`chainerx::Device`](chainerx_cc/chainerx/device.h) interfaces.

## Making your script use ChainerX

In order to utilize `chainerx`, you first need to allocate your model on a ChainerX device using `Chain.to_device` or `Link.to_device`. These are new methods that have been introduced to replace `to_cpu` and `to_gpu`, extending device transfer to arbitrary devices.
Similarly, you have to transfer the data (`Variable`s) to the same device before feeding them to the model.

Note that no breaking changes should have been introduced and any existing Chainer code (that works with the current master branch of Chainer) is expected to work.

### Will my `FunctionNode` work with ChainerX?

It will not break because of the fallback mechanism explained above, but you will not see any performance improvements (but most likely a degradation because of the additional conversions).

To support ChainerX with your `FunctionNode`, you need to implement `FunctionNode.forward_chainerx` with the same signature as `FunctionNode.forward`, but where given inputs are of type `chainerx.ndarray`. It is expected to return a `tuple` just like `FunctionNode.forward`.

The example below shows how `matmul` is extended to support ChainerX. Note that `chainer.Fallback` can be returned in case the function is not supported by ChainerX or cannot be achieved by a combination of ChainerX functions. This is also the default behavior in case the method is not implemented at all.

```python
class MatMul(function_node.FunctionNode):
    ...

    def forward_chainerx(self, x):
        a, b = x
        if self.transa or self.transb or self.transc:
            return chainer.Fallback
        if a.dtype != b.dtype:
            return chainer.Fallback
        if a.ndim != 2 or b.ndim != 2:
            return chainer.Fallback
        if self.dtype is not None and self.dtype != a.dtype:
            return chainer.Fallback
        return chainerx.dot(a, b),  # Fast C++ implementation
    ...
```

### Example: MNIST

The [train_mnist.py example](examples/mnist/train_mnist.py) now runs with ChainerX.

## Known Issues

- Mixed dtypes are, in general, not supported. `FunctionNode`s will fall back to their NumPy/CuPy propagations.
- `chainer.Function` is not supported.
- ideep is not supported.

## FAQ

### Can I use ChainerX without Chainer?

Yes, it is possible. See the code samples below.

- [Train an MLP with MNIST](chainerx_cc/examples/mnist)
- [Train a CNN with ImageNet](chainerx_cc/examples/imagenet)
- [`chainerx` basic usage examples](tests/chainerx_tests/acceptance_tests)

### What does the C++ interface look like?

It is almost identical to the Python interface with a 1-to-1 mapping.
The binding layer is thin and many of the types defined in C++ have Python equivalent classes.
The bindings are defined [here](https://github.com/pfnet/chainerx/tree/master/chainerx_cc/chainerx/python).
