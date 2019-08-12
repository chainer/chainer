import typing as tp  # NOQA

from chainer import link_hook


class TweakLink(link_hook.LinkHook):

    """LinkHook abstraction to manipulate any attributes of the \
:class:`~chainer.Link`.

    This class is aimed at easing to tweak the added
    links attributes including its parameters and persistents (a.k.a.
    buffers or states).
    For example, masking and normalizing
    the weight of added link like `Masked AutoEncoder for Density Estimation
    <https://arxiv.org/abs/1502.03509>`_ (MADE) and
    :class:`~chainer.link_hooks.SpectralNormalization`.

    By design, you can edit attributes other than the weight of link such as
    bias and ``avg_mean`` and ``avg_var`` of
    :class:`~chainer.links.BatchNormalization` by specifying ``target_name``
    appropriately.

    .. rubric:: Required methods

    Each concrete class must at least override the following method.

    ``adjust_target(self, cb_args)``
        Implements the desired manipulation to the target of
        :class:`~chainer.Link` specified by ``target_name`` and/or
        inputs for the link.
        This method is expected to return an object of the same type and shape
        of the target with ``target_name`` attribute.
        Even if the target of ``target_name`` attribute is intact,
        return it as is for the consistency. Therefore, the simplest
        implementation of this method is as below:

        .. code-block:: python

            ...

            def adjust_target(self, cb_args):
                return getattr(cb_args.link, self.target_name)

            ...

    .. rubric:: Optional methods

    ``prepare_parameters(self, link)``
        Defines and registers states of a concrete class.
        Foe example, a scaling and shifting parameters unique to it.
        If you register :class:`~chainer.Parameter`\\s
        or :ref:`ndarray`\\s to the link, you need to delete them
        in :meth:`deleted`. See
        :class:`~chainer.link_hooks.SpectralNormalization` as an example.

    ``deleted(self, link)``
        Deletes all the objects you registered to the link via this hook if
        :meth:`prepare_parameters` is overridden.

    .. admonition:: Example

        1. Scaling the weight randomly if it's called for the even number
           of times.

            .. code-block:: python

                import random

                from chainer import link_hooks


                class RandomWeightScaling(link_hooks.TweakLink):

                   name = "RandomWeightScaling"

                   def __init__(self):
                       self.n = 0
                       super(RandomWeightScaling, self)__init__()

                   def adjust_target(self, cb_args):
                       link = cb_args.link
                       weight = getattr(link, self.target_name)
                       if self.n % 2 == 0:
                           return weight * random.random()
                       return weight

        2. Masking the weight of :class:`~chainer.links.Linear`

            .. code-block:: python

                from chainer import link_hooks


                class MaskingLinearRandomly(link_hooks.TweakLink):

                    name = "MaskingLinearRandomly"

                    def adjust_target(self, cb_args):
                        link = cb_args.link
                        weight = getattr(link, self.target_name)
                        mask = link.xp.random.randint(
                            0, 2, weight.shape, link.xp.uint8)
                        return weight * mask

        3. Perturb the input using the statistics of the weight.

            .. code-block:: python

                from chainer import link_hooks


                class InputsPerturbation(link_hooks.TweakLink):

                    name = "InputsPerturbation"

                    def __init__(self, scale=0.001):
                        self.scale = scale
                        super(InputsPerturbation, self).__init__()

                    def adjust_target(self, cb_args):
                        link = cb_args.link
                        weight = getattr(link, self.target_name)
                        mu = link.xp.mean(weight.array)
                        for x in cb_args.args:
                            x += cb_args.link.xp.random.uniform(
                                -self.scale, self.scale, x.shape
                            ).astype(x.dtype) * mu
                        return weight

    Args:
        target_name (str): The name of the link's attribute which this hook
            manipulate. The default value is 'W'.
        name (str): If not ``None``, override the name of this hook.
            The default value is ``None``.
        axis (int): The axis of the target attribute representing the size of
            output feature. The default value is ``None``.

    """

    name = 'TweakLink'

    def __init__(self, target_name='W', name=None, axis=None):
        # type: (str, tp.Optional[str], tp.Optional[int]) -> None
        self.target_name = target_name
        if name is not None:
            self.name = name
        if axis is not None:
            self.axis = axis
        self.is_link_initialized = False

    def __enter__(self):
        """TweakLink and its concrete classes cannot be a context manager."""
        raise NotImplementedError(
            '"%s" cannot be a context manager.' % self.name)

    def __exit__(self):
        """TweakLink and its concrete classes cannot be a context manager."""
        raise NotImplementedError(
            '"%s" cannot be a context manager.' % self.name)

    def added(self, link):
        # type: ('chainer.link.Link') -> None
        """Callback function invoked when the link hook is registered.

        Args:
             link (~chainer.Link): Link object to which
                the link hook is registered.

        """
        if not hasattr(link, self.target_name):
            raise ValueError(
                'Weight of \'%s\' does not exist' % self.target_name)
        if getattr(link, self.target_name).array is not None:
            self.is_link_initialized = True
            self.prepare_parameters(link)

    def deleted(self, link):
        # type: ('chainer.link.Link') -> None
        """Callback function invoked when the link hook is unregistered.

        This callback is supposed to be overridden if this hook add additional
        parameters to the given link.

        Args:
            link (~chainer.Link): Link object to which
                the link hook is unregistered.
        """
        pass

    def forward_preprocess(self, cb_args):
        # type: (_ForwardPreprocessCallbackArgs) -> None
        """Callback function invoked before a forward call of a link.

        Basically, this function does not need to be overridden.

        Args:
            args: Callback data. It has the following attributes:

                * link (:class:`~chainer.Link`)
                    Link object.
                * forward_name (:class:`str`)
                    Name of the forward method.
                * args (:class:`tuple`)
                    Non-keyword arguments given to the forward method.
                * kwargs (:class:`dict`)
                    Keyword arguments given to the forward method.
        """
        if not self.is_link_initialized:
            # We need to initialize the link before `link.forward`
            # that usually initializes the link is called.
            self.initialize_link(cb_args)
            self.prepare_parameters(cb_args.link)
        self.original_target = getattr(cb_args.link, self.target_name)
        # Modify the target and set it as an attribute.
        adjusted_target = self.adjust_target(cb_args)
        setattr(cb_args.link, self.target_name, adjusted_target)

    def forward_postprocess(self, cb_args):
        # type: (_ForwardPostprocessCallbackArgs) -> None
        """Callback function invoked after a forward call of a link.

        This method also does not need to be overridden like
        :meth:`forward_preprocess`.

        Args:
            args: Callback data. It has the following attributes:

                * link (:class:`~chainer.Link`)
                    Link object.
                * forward_name (:class:`str`)
                    Name of the forward method.
                * args (:class:`tuple`)
                    Non-keyword arguments given to the forward method.
                * kwargs (:class:`dict`)
                    Keyword arguments given to the forward method.
                * out
                    Return value of the forward method.
        """
        # Here, the computational graph is already created,
        # we can reset the target to the original.
        setattr(cb_args.link, self.target_name, self.original_target)

    def adjust_target(self, cb_args):
        # type: (_ForwardPreprocessCallbackArgs) -> 'chainer.variable.Variable'  # NOQA
        """Adjust the target.

        This method implements the processing you want to apply to the link.

        Args:
            args: Callback data. It has the following attributes:

                * link (:class:`~chainer.Link`)
                    Link object.
                * forward_name (:class:`str`)
                    Name of the forward method.
                * args (:class:`tuple`)
                    Non-keyword arguments given to the forward method.
                * kwargs (:class:`dict`)
                    Keyword arguments given to the forward method.
        """
        raise NotImplementedError()

    def initialize_link(self, cb_args):
        # type: (_ForwardPreprocessCallbackArgs) -> 'chainer.variable.Variable'  # NOQA
        """Initialize the link if it is not in its declaration."""
        link = cb_args.link
        inputs = cb_args.args
        if not hasattr(link, '_initialize_params'):
            raise ValueError(
                'Link cannot be initialized by "%s"' % self.name)
        x = inputs[0]
        link._initialize_params(x.shape[1])
        self.is_link_initialized = True

    def prepare_parameters(self, link):
        # type: ('chainer.link.Link') -> None
        """Prepare parameters and persistents if necessary."""
        pass
