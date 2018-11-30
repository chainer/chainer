import unittest

import chainer
from chainer import testing


class TestBackwardCompatibilityAliases(unittest.TestCase):

    def test_aliases(self):
        # assert <alias path> is <genuine path>
        assert chainer.cuda is chainer.backends.cuda
        assert (chainer.should_use_cudnn
                is chainer.backends.cuda.should_use_cudnn)
        assert (chainer.should_use_cudnn_tensor_core
                is chainer.backends.cuda.should_use_cudnn_tensor_core)
        assert chainer.function.FunctionHook is chainer.FunctionHook
        assert (chainer.training.trigger._never_fire_trigger
                is chainer.training.util._never_fire_trigger)
        assert (chainer.training.trigger.get_trigger
                is chainer.training.get_trigger)
        assert (chainer.training.trigger.IntervalTrigger
                is chainer.training.triggers.IntervalTrigger)
        assert (chainer.training.updater.StandardUpdater
                is chainer.training.updaters.StandardUpdater)
        assert (chainer.training.updater.ParallelUpdater
                is chainer.training.updaters.ParallelUpdater)
        assert (chainer.training.extensions.dump_graph
                is chainer.training.extensions.computational_graph.DumpGraph)


testing.run_module(__name__, __file__)
