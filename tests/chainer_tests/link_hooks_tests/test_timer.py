import re
import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import link_hooks
from chainer import testing
from chainer.testing import attr


class MyModel(chainer.Chain):

    def __init__(self):
        super(MyModel, self).__init__()
        with self.init_scope():
            self.l1 = chainer.links.Linear(2, 3)
            self.l2 = chainer.links.Linear(3, 4)

    def forward(self, x):
        h = self.l1(x)
        return self.l2(h)


class TestTimerHook(unittest.TestCase):

    def test_name(self):
        assert link_hooks.TimerHook().name == 'TimerHook'

    def check_forward(self, xp):
        link = MyModel()
        hook = link_hooks.TimerHook()

        with hook:
            link(chainer.Variable(xp.array([[7, 5]], numpy.float32)))
            link(chainer.Variable(xp.array([[8, 1]], numpy.float32)))

        # Probing features

        def check_probing_features():

            # call_history
            hist = list(hook.call_history)
            assert len(hist) == 6
            names = [h.link_name for h in hist]
            times = [h.elapsed_time for h in hist]
            assert names == [
                'Linear', 'Linear', 'MyModel', 'Linear', 'Linear', 'MyModel']
            assert times[0] + times[1] < times[2]
            assert times[3] + times[4] < times[5]

            # total_time: Note that it's not the sum of all elapsed times
            assert hook.total_time() == times[2] + times[5]

            # summary
            summary = hook.summary()
            assert sorted(summary.keys()) == ['Linear', 'MyModel']
            assert summary['Linear'] == {
                'occurrence': 4,
                'elapsed_time': times[0] + times[1] + times[3] + times[4]}
            assert summary['MyModel'] == {
                'occurrence': 2,
                'elapsed_time': times[2] + times[5]}
            summary['Linear']

            # print_report
            s = six.StringIO()
            hook.print_report(file=s)
            report = s.getvalue().split('\n')
            assert len(report) == 4  # report[0] is the header line
            assert re.match(r' +MyModel +[.0-9a-z]+ +2$', report[1])
            assert re.match(r' +Linear +[.0-9a-z]+ +4$', report[2])
            assert len(report[3]) == 0

        # Probing the record should not change the internal state
        # (e.g. clearing the history).
        check_probing_features()
        check_probing_features()

    def test_forward_cpu(self):
        self.check_forward(numpy)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.cupy)


testing.run_module(__name__, __file__)
