from chainer.training import extension


class NaNKiller(extension.Extension):

    def __call__(self, trainer):
        optimizers = trainer.updater.get_all_optimizers()
        for optimizer in optimizers.values():
            target = optimizer.target
            xp = target.xp
            for param in target.params():
                if xp.isnan(param.array).any():
                    raise RuntimeError('NaN detected. R.I.P.')
