import chainer
import chainer.links.model


class DataParallelOptimizerCumulateGradientsHook(object):
    """
    A hook which sums up all replication's gradients in a
    DataParallel-Scenario
    """

    name = "DataParallelCumulateGradients"
    call_for_each_param = False
    timing = 'pre'

    def __call__(self, optimizer):
        """
        Summing up all parameters if the target is an instance of
        ~chainer.links.model.DataParallel

        Args
            optimizer (~chainer.Optimizer):
                the optimizer holding the target, whose gradients should be
                summed across the replications

        """
        if isinstance(optimizer.target, chainer.links.model.DataParallel):
            for module in optimizer.target.modules[1:]:
                optimizer.target.modules[0].addgrads(module)


class DataParallelOptimizerUpdateModelParameters(object):
    """
    A hook to replicate all parameters from the root model, to all
    model-replicas after the optimizer step
    """

    name = "DataParallelUpdateModelParams"
    call_for_each_param = False
    timing = "post"

    def __call__(self, optimizer):
        """
        Copying all parameters across the model replicas after optimizer update

        Args
            optimizer (~chainer.Optimizer):
                the optimizer holding the target, whose parameters should be
                updated across the replications

        """
        if isinstance(optimizer.target, chainer.links.model.DataParallel):
            for module in optimizer.target.modules[1:]:
                module.copyparams(optimizer.target.modules[0])
