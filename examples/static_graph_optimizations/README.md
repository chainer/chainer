
#### Demonstration of  "static subgraph optimizations" feature

#### Examples

The subfolders in this directory contain versions of the following Chainer
models that were modified to support this feature. Note that the MNIST
and CIFAR examples are mostly unchanged, except for the addition of the
`@static_graph` decorator to the model chain. The ptb example was
modified so that the RNN is explicitly unrolled inside of a static
chain.

* MNIST

* CIFAR

* ptb

You may find it interesting to try running the examples both with and
without static subgraph optimizations (i.e., by commenting out the
decorator) to see the effect on runtime performance and memory usage.
Note that models such as the CIFAR example in which the main performance
bottleneck is in computing the convolution functions, may not benefit
from this optimization. However, other models such as RNNs (ptb) may
show a more significant speedup.