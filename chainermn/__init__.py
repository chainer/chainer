import chainer

from chainermn import communicators  # NOQA
from chainermn import datasets  # NOQA
from chainermn import extensions  # NOQA
from chainermn import functions  # NOQA
from chainermn import global_except_hook  # NOQA
from chainermn import iterators  # NOQA
from chainermn import links  # NOQA
from chainermn import optimizers  # NOQA

from chainermn.communicators import CommunicatorBase  # NOQA
from chainermn.communicators import create_communicator  # NOQA
from chainermn.datasets import DataSizeError  # NOQA
from chainermn.datasets import scatter_dataset  # NOQA
from chainermn.extensions import create_multi_node_checkpointer  # NOQA
from chainermn.extensions import create_multi_node_evaluator  # NOQA
from chainermn.links import MultiNodeChainList  # NOQA
from chainermn.optimizers import create_multi_node_optimizer  # NOQA

global_except_hook._add_hook_if_enabled()

__version__ = chainer.__version__
