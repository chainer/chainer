import os
import unittest

from examples.tests import runner


EXAMPLES_ROOT = runner.EXAMPLES_ROOT


class TestImagenet(unittest.TestCase):
    def test_1(self):
        root_dir = os.path.join(EXAMPLES_ROOT, 'imagenet')
        image_root_dir = os.path.join(root_dir, '.testdata/images')
        list_file = os.path.join(root_dir, '.testdata/data.txt')

        with runner.ExampleRunner(root_dir) as r:

            r.run(
                'compute_mean.py',
                [
                    '-R', image_root_dir,
                    list_file,
                ])

            r.run(
                'train_imagenet.py',
                [
                    '-a', 'nin', '-R', image_root_dir,
                    '-B', '1', '-b', '1', '-E', '1',
                    list_file,
                    list_file,
                ])
