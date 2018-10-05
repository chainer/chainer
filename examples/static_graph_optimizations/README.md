
#### Demonstration of  "static subgraph optimizations" feature

The subfolders in this directory contain versions of the following Chainer
examples that were modified to support this feature. Since the static
graph optimizations feature is experimental, these examples might
be unstable.

* MNIST: This example includes two models: A version in which
`@static_graph` was added to the model chain, and a version that
demonstrates the use of "side effect" code.

* CIFAR: This is mostly unchanged from the existing CIFAR example except
that `@static_graph` was added to the model chain.

* ptb: This example was
modified so that the RNN is explicitly unrolled inside of a static
chain. This was necessary because the `LSTM` link is only partially
compatible with this feature.

You may find it interesting to run and time the examples both with and
without static subgraph optimizations (i.e., by commenting out the
decorator) to see the effect on runtime performance and memory usage.

Depending on the model and other hyperparameters such as the batch size,
enabling graph optimizations might not always result in a speedup. An
example of this is the CIFAR example, which does not run any faster
with this feature enabled. This is because GPU kernels such
as the convolution and multiplication operations already account
for most of the execution time. As a general rule, if the GPU
utilization is already close to 100%, the model is unlikely to
benefit from this feature.
