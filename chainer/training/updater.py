class Updater(object):

    """Interface of updater objects for trainers.

    TODO(beam2d): document it.

    """

    def connect_trainer(self, trainer):
        """Connects the updater to the trainer that will call it.

        The typical usage of this method is to register additional links to the
        reporter of the trainer. This method is called at the end of the
        initialization of :class:`~chainer.training.Trainer`. The default
        implementation does nothing.

        Args:
            trainer (~chainer.training.Trainer): Trainer object to which the
                updater is registered.

        """
        pass

    def finalize(self):
        """Finalizes the updater object.

        This method is called at the end of training loops. It should finalize
        each dataset iterator used in this updater.

        """
        raise NotImplementedError

    def get_optimizer(self, name):
        """Gets the optimizer of given name.

        Updater holds one or more optimizers with names. They can be retrieved
        by this method.

        Args:
            name (str): Name of the optimizer.

        Returns:
            ~chainer.Optimizer: Optimizer of the name.

        """
        raise NotImplementedError

    def get_all_optimizers(self):
        """Gets a dictionary of all optimizers for this updater.

        Returns:
            dict: Dictionary that maps names to optimizers.

        """
        raise NotImplementedError

    def update(self):
        """Updates the parameters of the target model.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.

        This method is called once at each iteration of the training loop.

        """
        raise NotImplementedError

    def serialize(self, serializer):
        """Serializes the current state of the updater object."""
        raise NotImplementedError


# For backward compatibility
from chainer.training.updaters.parallel_updater import ParallelUpdater  # NOQA
from chainer.training.updaters.standard_updater import StandardUpdater  # NOQA
