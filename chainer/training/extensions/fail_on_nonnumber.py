from chainer.training import extension


class FailOnNonNumber(extension.Extension):
    """Trainer extension to raise RuntimeError if parameters contain NaN or Inf.

    Although parameters including non-number such as NaN and Inf are
    unnecessary in most cases, :class:`~chainer.training.Trainer` will continue
    to compute even if the parameters in a given optimizer diverge.
    This extension is aimed to reduce unnecessary computations by throwing
    ``RuntimeError`` if the parameters contain NaN or Inf.
    """

    def __call__(self, trainer):
        optimizers = trainer.updater.get_all_optimizers()
        for name, optimizer in optimizers.items():
            target = optimizer.target
            xp = target.xp
            for param in target.params():
                if not xp.isfinite(param.array).all():
                    raise RuntimeError(
                        'Kill the process since parameters in optimizer'
                        ' \'{}\' diverge. R.I.P.'.format(name))
