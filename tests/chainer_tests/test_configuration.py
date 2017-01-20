import io
import threading
import unittest

import chainer
from chainer import configuration
from chainer import testing


class TestLocalConfig(unittest.TestCase):

    def setUp(self):
        self.global_config = configuration.GlobalConfig()
        self.config = configuration.LocalConfig(self.global_config)

        self.global_config.x = 'global x'
        self.global_config.y = 'global y'
        self.config.y = 'local y'
        self.config.z = 'local z'

    def test_attr(self):
        self.assertTrue(hasattr(self.config, 'x'))
        self.assertEqual(self.config.x, 'global x')
        self.assertTrue(hasattr(self.config, 'y'))
        self.assertEqual(self.config.y, 'local y')
        self.assertTrue(hasattr(self.config, 'z'))
        self.assertEqual(self.config.z, 'local z')
        self.assertFalse(hasattr(self.config, 'w'))

        del self.config.y
        self.assertTrue(hasattr(self.config, 'y'))
        self.assertEqual(self.config.y, 'global y')

        with self.assertRaises(AttributeError):
            del self.config.x

    def test_multi_thread_attr(self):
        def target():
            self.config.y = 'local y2'
            self.global_config.x = 'global x2'
            self.global_config.z = 'global z2'

        thread = threading.Thread(target=target)
        thread.start()
        thread.join()

        self.assertEqual(self.config.y, 'local y')
        self.assertEqual(self.config.x, 'global x2')
        self.assertEqual(self.config.z, 'local z')
        self.assertEqual(self.global_config.z, 'global z2')

    def test_using_config_local_did_not_exist(self):
        with chainer.using_config('x', 'temporary x', self.config):
            self.assertEqual(self.config.x, 'temporary x')
            self.assertEqual(self.global_config.x, 'global x')
        self.assertEqual(self.config.x, 'global x')
        self.global_config.x = 'global x2'
        self.assertEqual(self.config.x, 'global x2')

    def test_using_config_local_existed(self):
        with chainer.using_config('y', 'temporary y', self.config):
            self.assertEqual(self.config.y, 'temporary y')
            self.assertEqual(self.global_config.y, 'global y')
        self.assertEqual(self.config.y, 'local y')

    def test_print_config(self):
        sio = io.StringIO()
        self.config.show(sio)
        contents = sio.getvalue()
        self.assertEqual(contents, 'x:\tglobal x\ny:\tlocal y\nz:\tlocal z\n')


testing.run_module(__name__, __file__)
