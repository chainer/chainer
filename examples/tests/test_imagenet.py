import os
import unittest

from examples.tests import runner


EXAMPLES_ROOT = runner.EXAMPLES_ROOT


class TestImagenet(unittest.TestCase):
    def test_1(self):
        image_root_dir = os.path.join(
            EXAMPLES_ROOT, 'imagenet/.testdata/images')
        list_file = os.path.join(
            EXAMPLES_ROOT, 'imagenet/.testdata/data.txt')
        template_dir = os.path.join(
            EXAMPLES_ROOT, 'imagenet/.testdata/templates')

        with runner.ExampleRunner(template_dir=template_dir) as r:

            r.run(
                os.path.join(EXAMPLES_ROOT, 'imagenet/compute_mean.py'),
                [
                    '-R', image_root_dir,
                    list_file,
                ])

            r.run(
                os.path.join(EXAMPLES_ROOT, 'imagenet/train_imagenet.py'),
                [
                    '-a', 'nin', '-R', image_root_dir,
                    '-B', '1', '-b', '1', '-E', '1',
                    list_file,
                    list_file,
                ])
