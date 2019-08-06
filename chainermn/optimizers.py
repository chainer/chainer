import chainer
import copy


class _MultiNodeOptimizer(object):

    def __init__(self, actual_optimizer, communicator, zero_fill):
        super(_MultiNodeOptimizer, self).__setattr__(
            'communicator', communicator)
        super(_MultiNodeOptimizer, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_MultiNodeOptimizer, self).__setattr__(
            'target_params', [])
        super(_MultiNodeOptimizer, self).__setattr__(
            'zero_fill', zero_fill)

    def update(self, lossfun=None, *args, **kwds):
        target = self.target
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                target.cleargrads()
            else:
                target.zerograds()
            loss.backward(loss_scale=self.actual_optimizer._loss_scale)
            del loss

        if self.is_changed(target):
            self.communicator.bcast_data(target)
        else:
            self.communicator.multi_node_mean_grad(target, self.zero_fill)
            self.actual_optimizer.update(None, *args, **kwds)

    def is_changed(self, target):
        previous_params = self.target_params
        super(_MultiNodeOptimizer, self).__setattr__(
            'target_params', [(name, param.data is not None)
                              for name, param in sorted(target.namedparams())])
        if len(previous_params) != len(self.target_params):
            return True

        for param1, param2 in zip(self.target_params, previous_params):
            if (param1[0] != param2[0]) or param1[1] != param2[1]:
                return True
        return False

    def setup(self, link):
        self.actual_optimizer.setup(link)
        return self

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)


class _DoubleBufferingOptimizer(object):

    def __init__(self, actual_optimizer, communicator, zero_fill):
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'communicator', communicator)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'actual_optimizer', actual_optimizer)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'needs_update', False)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'communicated_target', None)
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'target_params_list', [[], []])
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'allreduce_grad_stream', chainer.cuda.Stream(non_blocking=True))
        super(_DoubleBufferingOptimizer, self).__setattr__(
            'zero_fill', zero_fill)

    def update(self, lossfun=None, *args, **kwds):
        target = self.target
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                target.cleargrads()
            else:
                target.zerograds()
            loss.backward(loss_scale=self.actual_optimizer._loss_scale)
            del loss

        if self.is_changed(target, self.target_params_list[0]):
            self.wait()
            self.communicator.bcast_data(target)
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'communicated_target', copy.deepcopy(target))
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'target_params_list', [
                    list(sorted(self.target.namedparams())),
                    list(sorted(self.communicated_target.namedparams()))])
            super(_DoubleBufferingOptimizer, self).__setattr__(
                'needs_update', False)
        else:
            self.wait()
            self.swap_grad(self.target_params_list[0],
                           self.target_params_list[1])
            self.multi_node_mean_grad_async()
            if self.needs_update:
                self.actual_optimizer.update(None, *args, **kwds)
            else:
                super(_DoubleBufferingOptimizer, self).__setattr__(
                    'needs_update', True)

    def multi_node_mean_grad_async(self):
        self.communicator._multi_node_mean_grad_async(
            self.communicated_target, self.zero_fill,
            self.allreduce_grad_stream)

    def is_changed(self, target, previous_params):
        target_params = list(sorted(target.namedparams()))
        if len(previous_params) != len(target_params):
            return True

        for param1, param2 in zip(target_params, previous_params):
            name1, var1 = param1
            name2, var2 = param2
            if (name1 != name2) or (var1.data is None) != (var2.data is None):
                return True
        return False

    def swap_grad(self, target1_params, target2_params):
        for param1, param2 in zip(target1_params, target2_params):
            _, var1 = param1
            _, var2 = param2
            var1.grad, var2.grad = var2.grad, var1.grad

    def wait(self):
        self.allreduce_grad_stream.synchronize()
        chainer.cuda.Stream.null.synchronize()

    def setup(self, link):
        self.actual_optimizer.setup(link)
        return self

    def __getattr__(self, attr_name):
        return getattr(self.actual_optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_optimizer, attr_name, value)


def create_multi_node_optimizer(actual_optimizer, communicator,
                                double_buffering=False, zero_fill=True):
    """Create a multi node optimizer from a Chainer optimizer.

    Args:
        actual_optimizer: Chainer optimizer
            (e.g., ``chainer.optimizers.Adam``).
        communicator: ChainerMN communicator.
        double_buffering: If ``True``, all-reduce and other
             processing (such as forward and backward) are
             overlapped using double buffering.
             There are cases where accuracy is affected because
             the gradients of the previous iteration are used
             for update. This flag is supported by
             ``PureNcclCommunicator`` only.
        zero_fill: A knob to control whether to fill gradients of initialized
             and unused Link (which is None internally) with zero-valued array,
             because the all gradients must be an array among processes for
             performing all-reduce, which might be an array or None after
             backward computation. Gradients of uninitialized Link are skipped.
             If it is False, gradients of unused Link are just skipped.
    Returns:
        The multi node optimizer based on ``actual_optimizer``.
    """
    if double_buffering:
        from chainermn.communicators.pure_nccl_communicator \
            import PureNcclCommunicator
        if not isinstance(communicator, PureNcclCommunicator):
            raise ValueError(
                'This communicator does not support double buffering.')
        return _DoubleBufferingOptimizer(actual_optimizer, communicator,
                                         zero_fill)
    return _MultiNodeOptimizer(actual_optimizer, communicator,
                               zero_fill)
