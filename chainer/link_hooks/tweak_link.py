from chainer import link_hook


class TweakLink(link_hook.LinkHook):

    """LinkHook abstraction to manipulate any attributes of the \
:class:`~chainer.link.Link`.

    This class is aimed at easing to tweak the added
    :class:`~chainer.link.Link` attributes including
    its parameters and persistents. For example, masking and normalizing
    the weight of added :class:`~chainer.link.Link` like
    `Masked AutoEncoder for Density Estimation
    <https://arxiv.org/abs/1502.03509>`_ (MADE) and
    :class:`~chainer.link_hooks.SpectralNormalization`.

    .. rubric:: Required methods

    Each concrete class must at least override the following method.

    ``adjust_target(self, cb_args)``
        Implements the desired manipulation to the target of
        :class:`~chainer.link.Link` specified by `target_name` and/or
        inputs for the link.
        This method is expected to return an object of the same type and shape
        of the the target with `target_name` attribute.
        Even if you do not touch the target of `target_name` attribute,
        return it as is for the consistency. Therefore, the simplest
        implementation of this method is just returning the target, i.e.,
        `return getattr(cb_args.link, self.target_name)`.

    .. rubric:: Optional methods

    ``prepare_params(self, link)``
        Defines states of a concrete class. Foe example, a scaling and shifting
        parameters unique to it. If you register :class:`~chainer.Parameter`\\s
        or :ref:`ndarray`\\s to the link, you need to delete them
        when :meth:`deleted` is called.
        See.. :class:`~chainer.link_hooks.SpectralNormalization`.

    .. admonition:: Example

        1. Scaling the weight randomly if it's called for the even number
           of times.

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
            manipulate. The default value is `'W'`.
        name (str): If not ``Nonee``, override the name of this hook.
            The default value is ``None``.
        axis (int): The axis of the target attribute representing the size of
            output feature. The default value is ``None``.

    """

    name = 'TweakLink'

    def __init__(self, target_name='W', name=None, axis=None, **kwargs):
        self.target_name = target_name
        if name is not None:
            self.name = name
        if axis is not None:
            self.axis = axis
        self.is_link_initialized = False

    def __enter__(self):
        raise NotImplementedError(
            '"{}" is not supposed to be used as a context manager.'.format(
                self.name))

    def __exit__(self):
        raise NotImplementedError(
            '"{}" is not supposed to be used as a context manager.'.format(
                self.name))

    def added(self, link):
        if not hasattr(link, self.target_name):
            raise ValueError(
                'Weight \'{}\' does not exist'.format(self.target_name))
        if getattr(link, self.target_name).array is not None:
            self.is_link_initialized = True
            self.prepare_params(link)

    def deleted(self, link):
        pass

    def forward_preprocess(self, cb_args):
        if not self.is_link_initialized:
            # We need to initialize the link before `link.forward`
            # that basically initializes the link is called.
            self.initialize_link(cb_args)
            self.prepare_params(cb_args.link)
        # Normalize the target weight and set the normalized as
        # the target link's attribute
        self.original_target = getattr(cb_args.link, self.target_name)
        adjusted_target = self.adjust_target(cb_args)
        setattr(cb_args.link, self.target_name, adjusted_target)

    def forward_postprocess(self, cb_args):
        setattr(cb_args.link, self.target_name, self.original_target)

    def adjust_target(self, cb_args):
        raise NotImplementedError()

    def initialize_link(self, cb_args):
        link = cb_args.link
        inputs = cb_args.args
        if not hasattr(link, '_initialize_params'):
            raise ValueError(
                'Link cannot be initialized by "{}"'.format(self.name))
        x = inputs[0]
        link._initialize_params(x.shape[1])
        self.is_link_initialized = True

    def prepare_params(self, link):
        pass
