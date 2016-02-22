import unittest

from chainer import trainer


class DoNothingExtension(trainer.Extension):

    def __call__(self, **kwargs):
        pass


class IncompleteExtension(trainer.Extension):
    pass


class TestExtension(unittest.TestCase):

    def test_default_attributes(self):
        ext = DoNothingExtension()
        self.assertEqual(ext.trigger, (1, 'iteration'))
        self.assertEqual(ext.priority, trainer.PRIORITY_READER)
        self.assertFalse(ext.invoke_before_training)
        self.assertEqual(ext.name, 'DoNothingExtension')

    def test_not_implemented_call(self):
        ext = IncompleteExtension()
        self.assertRaises(NotImplementedError, ext)


@trainer.make_extension()
def decorated_extension(**kwargs):
    pass


class TestMakeExtension(unittest.TestCase):

    def test_make_extension(self):
        @trainer.make_extension()
        def extension(**kwargs):
            pass

        self.assertEqual(extension.trigger, (1, 'iteration'))
        self.assertEqual(extension.priority, trainer.PRIORITY_READER)
        self.assertEqual(extension.invoke_before_training, False)
        self.assertFalse(hasattr(extension, 'name'))

    def test_make_extension_args(self):
        @trainer.make_extension(trigger=(1, 'epoch'), name='ext', priority=10,
                                invoke_before_training=True)
        def extension(**kwargs):
            pass

        self.assertEqual(extension.trigger, (1, 'epoch'))
        self.assertEqual(extension.name, 'ext')
        self.assertEqual(extension.priority, 10)
        self.assertTrue(extension.invoke_before_training)
