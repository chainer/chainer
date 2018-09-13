import re
import time
import unittest

import mock
import six

import chainer
from chainer import backend
from chainer import testing
from chainer import training
from chainer.training import extensions


def _get_mocked_trainer(links, stop_trigger=(10, 'iteration')):
    updater = mock.Mock()
    optimizer = mock.Mock()
    target = mock.Mock()
    target.namedlinks.return_value = [
        (str(i), link) for i, link in enumerate(links)]

    optimizer.target = target
    updater.get_all_optimizers.return_value = {'optimizer_name': optimizer}
    updater.iteration = 0
    updater.epoch = 0
    updater.epoch_detail = 0
    updater.is_new_epoch = True
    iter_per_epoch = 10

    def update():
        time.sleep(0.001)
        updater.iteration += 1
        updater.epoch = updater.iteration // iter_per_epoch
        updater.epoch_detail = updater.iteration / iter_per_epoch
        updater.is_new_epoch = updater.epoch == updater.epoch_detail

    updater.update = update

    return training.Trainer(updater, stop_trigger)


class TestParameterStatisticsBase(object):

    def setUp(self):
        self.trainer = _get_mocked_trainer(self.links)

    def create_extension(self, skip_statistics=False):
        kwargs = {
            'statistics': self.statistics if not skip_statistics else None,
            'report_params': self.report_params,
            'report_grads': self.report_grads,
            'prefix': self.prefix,
            'skip_nan_params': True  # avoid warnings when grads are nan
        }

        return extensions.ParameterStatistics(self.links, **kwargs)


@testing.parameterize(
    {
        'links': [chainer.links.Linear(3, 2)],
        'statistics': {'min': lambda x: backend.get_array_module(x).min(x)},
        'report_params': True,
        'report_grads': True,
        'prefix': None,
        'expect': 4
    },
    {
        'links': [chainer.links.Linear(3, 2)],
        'statistics': {'min': lambda x: backend.get_array_module(x).min(x)},
        'report_params': False,
        'report_grads': True,
        'prefix': 'test',
        'expect': 2
    },
    {
        'links': [chainer.links.Linear(3, 2)],
        'statistics': {'min': lambda x: backend.get_array_module(x).min(x)},
        'report_params': True,
        'report_grads': False,
        'prefix': None,
        'expect': 2
    },
    {
        'links': [chainer.links.Linear(3, 2)],
        'statistics': {'min': lambda x: backend.get_array_module(x).min(x)},
        'report_params': False,
        'report_grads': False,
        'prefix': 'test',
        'expect': 0
    },
    {
        'links': [chainer.links.Linear(3, 2)],
        'statistics': extensions.ParameterStatistics.default_statistics,
        'report_params': True,
        'report_grads': True,
        'prefix': None,
        'expect': 36
    },
    {
        'links': [chainer.links.Linear(3, 2)],
        'statistics': extensions.ParameterStatistics.default_statistics,
        'report_params': True,
        'report_grads': False,
        'prefix': 'test',
        'expect': 24
    },
    {
        'links': [chainer.links.Linear(3, 2)],
        'statistics': extensions.ParameterStatistics.default_statistics,
        'report_params': False,
        'report_grads': True,
        'prefix': None,
        'expect': 12
    },
    {
        'links': [chainer.links.Linear(3, 2)],
        'statistics': extensions.ParameterStatistics.default_statistics,
        'report_params': False,
        'report_grads': False,
        'prefix': 'test',
        'expect': 0
    }
)
class TestParameterStatistics(TestParameterStatisticsBase, unittest.TestCase):

    def test_report(self):
        self.trainer.extend(self.create_extension())
        self.trainer.run()
        self.assertEqual(len(self.trainer.observation), self.expect)

    def test_report_late_register(self):
        extension = self.create_extension(skip_statistics=True)
        for name, function in six.iteritems(self.statistics):
            extension.register_statistics(name, function)
        self.trainer.extend(extension)
        self.trainer.run()
        self.assertEqual(len(self.trainer.observation), self.expect)

    def test_report_key_pattern(self):
        self.trainer.extend(self.create_extension())
        self.trainer.run()

        pattern = r'^(.+/){2,}(data|grad)/.+[^/]$'
        for name in six.iterkeys(self.trainer.observation):
            if self.prefix is not None:
                assert name.startswith(self.prefix)

            match = re.match(pattern, name)
            assert match is not None
            if self.report_params and self.report_grads:
                pass
            elif self.report_params:
                assert 'data' == match.group(2)
            elif self.report_grads:
                assert 'grad' == match.group(2)


@testing.parameterize(
    {
        'links': [chainer.links.Linear(3, 2)],
        'statistics': {'one': lambda x: 1.0},
        'report_params': True,
        'report_grads': True,
        'prefix': 'test',
        'expect': 1.0
    }
)
class TestParameterStatisticsCustomFunction(TestParameterStatisticsBase,
                                            unittest.TestCase):

    def test_custom_function(self):
        extension = extensions.ParameterStatistics(
            self.links, statistics=self.statistics)
        self.trainer.extend(extension)
        self.trainer.run()

        for value in six.itervalues(self.trainer.observation):
            self.assertEqual(value, self.expect)


testing.run_module(__name__, __file__)
