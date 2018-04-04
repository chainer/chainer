from chainer.training import extension


class NaNKiller(extension.Extension):
    """Trainer extension to raise RuntimeError if parameters contain NaN.

    Although parameters including NaN are unnecessary in most cases,
    :class:`~chainer.training.Trainer` will continue to compute even if
    the parameters in a given optimizer diverge. This extension is aimed to
    reduce unnecessary computations by throwing ``RuntimeError``
    if the parameters contain NaN.
    """

    def __call__(self, trainer):
        optimizers = trainer.updater.get_all_optimizers()
        for optimizer in optimizers.values():
            target = optimizer.target
            xp = target.xp
            for param in target.params():
                if xp.isnan(param.array).any():
                    raise RuntimeError('NaN detected. R.I.P.')
