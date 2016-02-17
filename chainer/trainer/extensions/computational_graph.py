import os

from chainer import computational_graph
from chainer.trainer import extension


class ComputationalGraph(extension.Extension):

    """Trainer extension to dump a computational graph.

    TODO(beam2d): document it.

    """
    def __init__(self, target, varname='loss', outname='cg.dot'):
        self._done = False
        self.target = target
        self.varname = varname
        self.outname = outname

    def __call__(self, trainer):
        if self._done:
            return

        var = getattr(self.target, self.varname, None)
        if var is None:
            raise TypeError('computational graph target does not have the '
                            'specified attribute `{}\''.format(self.varname))
        cg = computational_graph.build_computational_graph([var]).dump()

        outpath = os.path.join(trainer.out, self.outname)
        # TODO(beam2d): support outputting to images by the dot command
        with open(outpath, 'w') as f:
            f.write(cg)

        self._done = True
