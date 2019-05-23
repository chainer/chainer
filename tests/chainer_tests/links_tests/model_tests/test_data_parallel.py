import unittest
import chainer
from chainer import testing
import chainer.link
import chainer.links
import chainer.functions


class TestDataParallel(unittest.TestCase):

    def setUp(self) -> None:

        # creating a really simple model to test dataparallel behavior
        class SimpleModel(chainer.link.Chain):
            def __init__(self):
                super(SimpleModel, self).__init__()

                with self.init_scope():
                    self.dense_1 = chainer.links.Linear(3, 32)
                    self.dense_2 = chainer.links.Linear(32, 2)

            def forward(self, x):
                return self.dense_2(chainer.functions.relu(self.dense_1(x)))

        self.model = chainer.links.DataParallel(SimpleModel(),
                                                devices=["@numpy", "@numpy"])

    # test with keyword arguments
    def test_keyword_arguments_different_batchsize(self):
        import numpy as np

        # test batchsize smaller than, equal to and greater than number devices
        for batchsize in [1, 2, 3]:
            with self.subTest(batchsize=batchsize):
                input_kwargs = {
                    "x": np.random.rand(batchsize, 3).astype(np.float32)
                }

                pred = self.model(**input_kwargs)
                self.assertTupleEqual(pred.shape,
                                      (batchsize, 2))
                self.assertEqual(chainer.get_device(pred.device),
                                 chainer.get_device("@numpy"))

    # test with positional arguments
    def test_positional_arguments(self):
        import numpy as np

        # test batchsize smaller than, equal to and greater than number devices
        for batchsize in [1, 2, 3]:
            with self.subTest(batchsize=batchsize):
                input_args = [
                    np.random.rand(batchsize, 3).astype(np.float32)
                ]

                pred = self.model(*input_args)
                self.assertTupleEqual(pred.shape,
                                      (batchsize, 2))

                self.assertEqual(chainer.get_device(pred.device),
                                 chainer.get_device("@numpy"))


testing.run_module(__name__, __file__)
