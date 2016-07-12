import copy

import six

from chainer import cuda
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import link
from chainer.training import extension
from chainer import variable


def concat_variables(variables, device=None):
    """Concatenates a list of variables into a single variable.

    The forward operation of a model yields a variable, tuple of variables
    or a dict of variables for each batch. This function concatenates a list
    of such variables to a single variable, tuple of varialbes or dict of
    variables, respectively.

    Args:
        variables (list): A list of variables, list of tuples of variables
            or a list of dicts of variables.
        device (int): Device ID to which each array is sent. Negative value
            indicates the host memory (CPU). If it is omitted, all arrays are
            left in the original device.

    Returns:
        Variable, a tuple of variables, or a dictionary of variables. The type
        depends on the type of each example in the variables list.

    """
    if len(variables) == 0:
        raise ValueError('No variables given')

    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        to_device = lambda x: cuda.to_gpu(x, device)

    first_elem = variables[0]

    if isinstance(first_elem, tuple):
        result = []
        for i in six.moves.range(len(first_elem)):
            result.append(to_device(_concat_variables(
                [example[i] for example in variables])))

        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}
        for key in first_elem:
            result[key] = to_device(_concat_variables(
                [example[key] for example in variables]))

        return result

    else:
        return to_device(_concat_variables(variables))


def _concat_variables(variables):
    xp = cuda.get_array_module(variables[0].data)
    with cuda.get_device(variables[0].data):
        return variable.Variable(
            xp.concatenate([var.data for var in variables])
        )


class Extractor(extension.Extension):

    """Trainer extension to extract features using a trained model.

    This extension extracts features using a trained model by a given
    extraction function.

    Extractor has a structure to customize similar to that of
    :class:`~chainer.training.extensions.Evaluator`.

    Args:
        iterator: Dataset iterator for extracting features. It can also be
            a dictionary of iterators. If this is just an iterator, the
            iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to process. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays.
            :func:`~chainer.dataset.concat_examples` is used by default.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        extract_hook: Function to prepare for each extraction process. It is
            called at the beginning of the extraction. The evxtractor extension
            object is passed at each call.
        extract_func: extraction function called at each iteration. The target
            link to extract as a callable is used by default.
        merger: Merger function to build output variable of features. By
            default it concatenates all output features along the batch axis.

    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        extract_hook: Function to prepare for each extraction process.
        extract_func: Extraction function called at each iteration.
        features: The extracted features. Created after calling the extension.
    """
    trigger = 1, 'epoch'
    default_name = 'extraction'
    priority = extension.PRIORITY_WRITER

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, extract_hook=None, extract_func=None,
                 merger=concat_variables):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target

        self.converter = converter
        self.merger = merger
        self.device = device
        self.extract_hook = extract_hook
        self.extract_func = extract_func
        self.features = None

    def get_iterator(self, name):
        """Returns the iterator of the given name."""
        return self._iterators[name]

    def get_all_iterators(self):
        """Returns a dictionary of all iterators."""
        return dict(self._iterators)

    def get_target(self, name):
        """Returns the target link of the given name."""
        return self._targets[name]

    def get_all_targets(self):
        """Returns a dictionary of all target links."""
        return dict(self._targets)

    def __call__(self, trainer=None):
        """Executes the extractor extension.

        Unlike usual extensions, this extension can be executed without passing
        a trainer object.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.

        """
        self.features = self.extract()
        return self.features

    def extract(self):
        """Extractes features using the target model.

        This method runs the extraction loop over the dataset. It accumulates
        the extracted features to a :class:`~chainer.Variable`.

        Users can override this method to customize the extraction routine.

        Returns:
            dict: Features variable.

        """
        iterator = self._iterators['main']
        target = self._targets['main']
        extract_func = self.extract_func or target

        if self.extract_hook:
            self.extract_hook(self)
        it = copy.copy(iterator)

        features = []
        for batch in it:
            in_arrays = self.converter(batch, self.device)
            if isinstance(in_arrays, tuple):
                in_vars = tuple(variable.Variable(x, volatile='on')
                                for x in in_arrays)
                features_batch = extract_func(*in_vars)
            elif isinstance(in_arrays, dict):
                in_vars = {key: variable.Variable(x, volatile='on')
                           for key, x in six.iteritems(in_arrays)}
                features_batch = extract_func(**in_vars)
            else:
                in_var = variable.Variable(in_arrays, volatile='on')
                features_batch = extract_func(in_var)

            features.append(features_batch)

        return self.merger(features)
