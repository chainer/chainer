import chainer.variable
import chainer.link


def _apply_scatter(inputs: chainer.variable.Variable, target_devices: list,
                   dim: int = 0):
    """
    Scatters inputs to target devices; Slicing will be done against a given
    dimension

    Args:
        inputs (~chainer.Variable): the input variable to scatter
        target_devices (list of str or ~chainer.backend.Device):
            the target devices to scatter to
        dim (int): the dimension to use for slicing
    Returns
        list of ~chainer.Variable: list of variable slices on correct devices

    """

    def _slice_inputs(input_var, dim, num_dims, start, end, target_device):
        """
        Slices the input variable along a given dimension from start to end
        and pushes it to correct device

        Args:
            input_var (~chainer.Variable): the variable to slice
            dim (int): the dimension to slice along
            num_dims (int): the dimensionality of ``input_var``
            start (int): the start value for slicing (included)
            end (int) : the end value for slicing (excluded)
            target_device (str or ~chainer.backend.Device):
                the device to push to
        Returns
            ~chainer.Variable: the slice of the variable

        """
        slc = [slice(None)] * num_dims
        slc[dim] = slice(start, end)
        sliced_var = input_var[slc]
        sliced_var.to_device(target_device)
        output_shape = list(input_var.shape)
        output_shape[dim] = -1
        return sliced_var.reshape(output_shape)

    # create empty sliced input list
    scattered_inputs = []

    # calculate constant only once
    num_devices = len(target_devices)
    samples_per_device = inputs.shape[dim] // num_devices
    num_dims = len(inputs.shape)

    # iterate over number of devices and slice accordingly
    # (exclude last device)
    # iterating until the minimum of num_devices and inputs.shape[dim] -1
    # ensures that if the batchsize is too small to be scattered across all
    # devices, we will only scatter across as many devices as possible
    for i in range(min(num_devices, inputs.shape[dim]) - 1):
        start, end = i * samples_per_device, i + 1 * samples_per_device
        scattered_inputs.append(_slice_inputs(inputs, dim,
                                              num_dims, start, end,
                                              target_devices[i]))

    # all remaining samples (not yet sliced) are appended now
    # (all samples used; will be pushed to last device later)
    scattered_inputs.append(_slice_inputs(
        inputs, dim, len(inputs.shape, ),
        (num_devices - 1) * samples_per_device,
        inputs.shape[dim], target_devices[-1]))

    return scattered_inputs


def _apply_gather(target_device, dim, *outputs):
    """
    Gathers all outputs from model replicas, pushes them to target_device
    and concatenates them along the given dimension

    Args:
         target_device (str or ~chainer.backend.Device):
            the target device to copy all outputs to
         dim (int): the dimension to use for concatenation
         outputs (list of ~chainer.Variable): the outputs obtained from model
         replicas

    Returns
        ~chainer.Variable: the concatenated outputs
    """
    for _output in outputs:
        _output.to_device(target_device)

    return chainer.functions.concat(outputs, dim)


def _scatter(inputs, target_devices: list, dim):
    """
    Scatters all inputs across given target_devices

    Args
        inputs (Any): the inputs to scatter across the devices
        target_devices (list of ~chainer.Variable): devices to scatter to
        dim (int): dimension to use for slicing

    Returns

    Any: iterable or mapping of scattered inputs (same type as inputs)

    """

    def _scatter_map(inputs):
        """
        Scatters all inputs across given target_devices. Supports arbitrary
        iterables and mappings (dicts)

        Args
            inputs (Any): The inputs to scatter

        Returns
            list of ~chainer.Variable: list of scattered inputs

        """

        # directly apply the scattering on variable
        if isinstance(inputs, chainer.variable.Variable):
            return _apply_scatter(inputs, target_devices, dim)

        # map _scatter_map recursively to all samples in tuple
        if isinstance(inputs, tuple) and len(inputs) > 0:
            return list(zip(*map(_scatter_map, inputs)))

        # map _scatter_map recursively to all samples in list
        if isinstance(inputs, list) and len(inputs) > 0:
            return list(map(list, zip(*map(_scatter_map,
                                           inputs))))

        # map _scatter_map recursively to all samples in dict
        if isinstance(inputs, dict) and len(inputs) > 0:
            return list(map(type(inputs), zip(*map(_scatter_map,
                                                   inputs.items()))))

        # try to convert inputs to chainer variable first and afterwards apply
        # _scatter_map again
        try:
            return _scatter_map(chainer.as_variable(inputs))
        except TypeError:
            return [inputs for targets in target_devices]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return _scatter_map(inputs)
    finally:
        _scatter_map = None


def _gather(outputs, target_device, dim=0):
    """
    Gathers tensors from different devices onto a single device

    Args
        outputs (Any): the outputs obtained from model replicas
        target_device (str or ~chainer.backend.Device):
            the target device to copy all outputs to
        dim (int): the dimension to use for concatenation

    """

    def gather_map(outputs):
        """
        Recursively gathers outputs from different devices

        Args
            outputs (Any): the outputs to gather

        Returns
            Any: the concatenated outputs (same type as outputs[0] before concat)
        """
        out = outputs[0]
        if isinstance(out, chainer.variable.Variable):
            return _apply_gather(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None


class DataParallel(chainer.link.Chain):
    """
    A Wrapper around a ~chainer.Chain instance to implement parallel
    training by splitting the batches
    """

    def __init__(self, module: chainer.link.Chain, devices: list,
                 batch_dim=0):
        """
        Args
            module (~chainer.Chain):
                the module to wrap (will be replicated on all devices)
            devices (list of str or ~chainer.backend.Device):
                a list containing the devices to use (either as strings or as
                ~chainer.backend.Device). The first device will be used as
                output device. Make sure, your labels are also on this device
                for loss calculation!
            batch_dim (int):
                the index of the batchdimension (usually 0, but can become
                e.g. 1 in NLP tasks)

        """
        super(DataParallel, self).__init__()

        modules = [module.copy() for _ in devices]

        for _module, _device in zip(modules, devices):
            _module.to_device(_device)

        with self.init_scope():
            self.modules = chainer.link.ChainList(*modules)

        self.devices = devices

        self._output_device = devices[0]
        assert self._output_device in self.devices
        self._output_device_idx = self.devices.index(self._output_device)
        self.dim = batch_dim

    def forward(self, *args, **kwargs):
        """
        Scatters the inputs (both positional and keyword arguments) across
        all devices, feeds them through model replicas and re-builds
        batches on output device

        Args
            args (tuple):
                positional arguments of arbitrary number and type
            kwargs (dict):
                keyword arguments of arbitrary number and type
        Returns
            Any:
                predictions (concatenation of all predictions obtained from
                model replicas)

        """
        scattered_args, scattered_kwargs = self._scatter(args, kwargs,
                                                         self.devices,
                                                         self.dim)

        predictions = []
        for _args, _kwargs, _module in zip(scattered_args, scattered_kwargs,
                                           self.modules):
            predictions.append(_module(*_args, *_kwargs))

        predictions = self._gather(predictions, self.dim,
                                   self._output_device)

        return predictions

    def params(self, include_uninit=True):
        """
        Only the parameters of the module on the first device will actually
        be updated, all the other parameters will be replicated by the
        optimizer after an update
        Parameters
        ----------
        include_uninit : bool
        Returns
        -------
        a generator holding the root-modules parameters
        """
        return self.modules[0].params(include_uninit)

    @staticmethod
    def _scatter(inputs, kwargs, target_devices: list, dim=0):
        """
        Scatters all inputs (args and kwargs) to target devices and splits
        along given dimension

        Args
            inputs (list or tuple):
                positional arguments
            kwargs (dict):
                keyword arguments
            target_devices (list of str or ~chainer.backend.Device):
                list of target device (either string or ~chainer.backend.Device)
            dim (int):
                the dimension, which should be used for splitting the batch

        Returns
            tuple: scattered positional arguments
            tuple: scattered keyword arguments

        """

        # scatter inputs if given
        inputs = _scatter(inputs, target_devices, dim) if inputs else []
        # scatter kwargs if given
        kwargs = _scatter(kwargs, target_devices, dim) if kwargs else []

        # extend lengths by empty tuples if necessary
        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])

        inputs = tuple(inputs)
        kwargs = tuple(kwargs)

        return inputs, kwargs

    @staticmethod
    def _gather(predictions, dim, target_device):
        """
        Re-Builds batches on the target device

        Args:
            predictions (list):
                list containing the predictions from all replicated models
            dim (int):
                dimension to use for concatenating single predictions
            target_device (str or ~chainer.backend.Device):
                the device, the re-built batch should lie on

        Returns
            Any: the rebuild batch (lying on target_device)

        """
        return _gather(predictions, target_device, dim)

    def zerograds(self):
        for module in self.modules:
            module.zerograds()

    def cleargrads(self):
        for module in self.modules:
            module.cleargrads()
