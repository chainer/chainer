from chainer import optimizer
import chainer.optimizer_hooks


class DataParallelOptimizer(optimizer.Optimizer):
    """
    An Optimizer-Wrapper to enable DataParallel. Basically this forwards
    all functions to the interal optimizer, but registers the additional
    hooks needed for DataParallel (namely
    ~chainer.optimizer_hooks.ParallelOptimizerUpdateModelParameters as a
    post-update hook and
    ~chainer.optimizer_hooks.ParallelOptimizerCumulateGradientsHook as a
    pre-update hook)

    Args
        optim (~chainer.Optimizer):
            the optimizer to wrap
    """

    def __init__(self, optim):

        if isinstance(optim, optimizer.Optimizer):
            self._optimizer = optim

        else:
            raise RuntimeError("Invalid optimizer class given: Expected "
                               "instance of chainer.Optimizer, but got %s"
                               % optim.__class__.__name__)

    @classmethod
    def from_optimizer_class(cls, optim_cls, *args, **kwargs):
        """
        Classmethod to create an instance given only the optimizer class and
        initialization arguments

        Args
            optim_cls (subclass of ~chainer.Optimizer):
                the optimizer to use internally
            args (tuple):
                arbitrary positional arguments (will be used for
                initialization of internally used optimizer)
            kwargs (dict):
                arbitrary keyword arguments (will be used for initialization
                of internally used optimizer)

        """
        if optim_cls is not None and issubclass(optim_cls,
                                                optimizer.Optimizer):
            _optim = optim_cls(*args, **kwargs)
        else:
            raise RuntimeError("Invalid optimizer class given: Expected "
                               "Subclass of chainer.Optimizer, but got %s"
                               % optim_cls.__name__)
        return cls(_optim)

    def setup(self, link):
        """
        Calls the setup method of the internal optimizer and registers the
        necessary grads for data-parallel behavior

        Args
            link (~chainer.links.models.DataParallel):
                the target, whose parameters should be updated

        """
        self._optimizer.setup(link)

        self._optimizer.add_hook(
            chainer.optimizer_hooks.DataParallelOptimizerCumulateGradientsHook(
            )
        )
        self._optimizer.add_hook(
            chainer.optimizer_hooks.DataParallelOptimizerUpdateModelParameters(
            )
        )

    # forward all functions and attributes to internal optimizer via properties
    @property
    def target(self):
        return self._optimizer.target

    @property
    def epoch(self):
        return self._optimizer.epoch

    @property
    def _pre_update_hooks(self):
        return self._optimizer._pre_update_hooks

    @property
    def _loss_scale(self):
        return self._optimizer._loss_scale

    @property
    def _loss_scale_max(self):
        return self._optimizer._loss_scale_max

    @property
    def _loss_scaling_is_dynamic(self):
        return self._optimizer._loss_scaling_is_dynamic

    @property
    def use_auto_new_epoch(self):
        return self._optimizer.use_auto_new_epoch

    @property
    def update(self):
        return self._optimizer.update

    @property
    def new_epoch(self):
        return self._optimizer.new_epoch

    @property
    def add_hook(self):
        return self._optimizer.add_hook

    @property
    def remove_hook(self):
        return self._optimizer.remove_hook

    @property
    def call_hooks(self):
        return self._optimizer.call_hooks

    @property
    def serialize(self):
        return self._optimizer.serialize

    @property
    def loss_scaling(self):
        return self._optimizer.loss_scaling

    @property
    def set_loss_scale(self):
        return self._optimizer.set_loss_scale

    @property
    def check_nan_in_grads(self):
        return self._optimizer.check_nan_in_grads

    @property
    def is_safe_to_update(self):
        return self._optimizer.is_safe_to_update

    @property
    def update_loss_scale(self):
        return self._optimizer.update_loss_scale
