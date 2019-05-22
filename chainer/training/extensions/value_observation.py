from chainer.training import extension


def observe_value(observation_key, target_func):
    """Returns a trainer extension to continuously record a value.

    Args:
        observation_key (str): Key of observation to record.
        target_func (function): Function that returns the value to record.
            It must take one argument: :class:~chainer.training.Trainer object.
    Returns:
        The extension function.
    """
    @extension.make_extension(
        trigger=(1, 'epoch'), priority=extension.PRIORITY_WRITER)
    def _observe_value(trainer):
        trainer.observation[observation_key] = target_func(trainer)
    return _observe_value


def observe_optimizer(
        hyperparam_name, optimizer_name='main', observation_key=None):
    """Returns a trainer extension to record a hyperparameter.

    Args:
        hyperparam_name (str): Name of hyperparameter.
        optimizer_name (str): Name of optimizer whose hyperparameter is
            recorded.
        observation_key (str): Key of observation to record.
            It is identical to ``hyperparam_name`` by default.

    Returns:
        The extension function.
    """

    if observation_key is None:
        observation_key = hyperparam_name

    def target_func(trainer):
        optimizer = trainer.updater.get_optimizer(optimizer_name)
        value = getattr(optimizer, hyperparam_name, None)
        if value is None:
            raise RuntimeError(
                'Hyperparameter not found in optimizer.\n'
                'Optimizer: {optname} ({opttype})\n'
                'Hyperparameter: {hyperparam}'.format(
                    optname=optimizer_name,
                    opttype=type(optimizer),
                    hyperparam=hyperparam_name))
        return value

    return observe_value(observation_key, target_func)


def observe_lr(optimizer_name='main', observation_key='lr'):
    """Returns a trainer extension to record the learning rate.

    Args:
        optimizer_name (str): Name of optimizer whose learning rate is
            recorded.
        observation_key (str): Key of observation to record.

    Returns:
        The extension function.

    .. note::
        :func:`~chainer.training.extensions.observe_optimizer` is preferable
        to ``observe_lr`` as the former is more general.

    """
    return observe_optimizer(
        'lr',
        optimizer_name=optimizer_name,
        observation_key=observation_key)
