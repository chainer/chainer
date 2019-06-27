from __future__ import absolute_import
import collections
import heapq
import warnings

import six

import chainer
from chainer import _backprop_utils
from chainer import backend
from chainer.utils import argument
import chainerx


def backward(outputs, grad_outputs=None, **kwargs):
    """backward(outputs, grad_outputs=None, *, enable_double_backprop=False)

    Runs backpropagation from variables simultaneously.

    .. warning::

        This feature is experimental. The interface can change in the future.

    Args:
        outputs (tuple or list of :class:`~chainer.Variable`):
            A sequence of output variables from which backprop starts.
        grad_outputs (None or tuple or list of :class:`~chainer.Variable`):
            A sequence of variables that gives the initial value of each output
            gradient.
            If this argument is ``None``, backprop uses
            :attr:`~chainer.Variable.grad_var` of ``outputs``.
        enable_double_backprop (bool): If ``True``,
            computational trace of the whole backpropagation procedure is
            recorded to the computational graph so that one can further do
            backpropagation from the resulting gradients. Note that
            enabling it results in larger memory consumption needed to
            store the gradients w.r.t intermediate variables that are
            required for the second gradient computation.

    .. seealso::
       :meth:`chainer.Variable.backward`
       :func:`chainer.grad`

    """
    enable_double_backprop, = argument.parse_kwargs(
        kwargs, ('enable_double_backprop', False),
        retain_grad='semantics for retain_grad=True is under discussion',
        loss_scale='chainer.backward does not support loss_scale option',
    )
    if not isinstance(outputs, (tuple, list)):
        raise TypeError(
            'outputs must be a tuple or a list, not {}.'.format(type(outputs)))
    for v in outputs:
        if not isinstance(v, chainer.Variable):
            raise TypeError(
                'each output must be a Variable, not {}'.format(type(v)))
    if grad_outputs is not None:
        if not isinstance(grad_outputs, (tuple, list)):
            raise TypeError(
                'grad_outputs must be None, a tuple, or a list, not {}.'
                .format(type(grad_outputs)))
        if len(outputs) != len(grad_outputs):
            raise ValueError(
                'grad_outputs must be of the same length as outputs.\n'
                'len(outputs) = {}, len(grad_outputs) = {}'
                .format(len(outputs), len(grad_outputs)))

    is_chainerx = [v._has_chainerx_array for v in outputs]

    if any(is_chainerx):
        if not all(is_chainerx):
            # The restriction is required as soon as the workarounds below
            # are removed.
            raise ValueError('cannot mix chainerx and other backends')

        # Cannot use chainerx.backward directly, because it does not follow
        # retain_grad=False
        # TODO(kataoka): Fix chainerx.backward and remove this workaround
        if grad_outputs is None:
            grad_outputs = []
            for y in outputs:
                grad_outputs.append(y.grad_var)
                y.grad_var = None

        # The check is required because chainerx.backward sets default grads.
        # TODO(kataoka): Fix chainerx.backward and remove this workaround
        indices = [i for i, gy in enumerate(grad_outputs) if gy is not None]
        outputs = [outputs[i] for i in indices]
        grad_outputs = [grad_outputs[i] for i in indices]

        # Use new variables to start backprop
        # TODO(kataoka): Implement chainerx.backward(output, grad_outputs)
        # and remove this workaround.
        outputs = chainer.functions.identity(*outputs)
        if not isinstance(outputs, tuple):
            outputs = outputs,
        grad_outputs = chainer.functions.identity(*grad_outputs)
        if not isinstance(grad_outputs, tuple):
            grad_outputs = grad_outputs,

        # TODO(kataoka): Even after F.identity, non-float grad cannot be set.
        # Move the check to elsewhere and remove this workaround.
        outputs_ = []
        for y, gy in zip(outputs, grad_outputs):
            if not y.requires_grad and gy is not None:
                warnings.warn(
                    'Some of grads are ignored by chainer.backward.\n'
                    'backend: ChainerX, '
                    'output.dtype: {}, grad_output.dtype: {}'.format(
                        y.dtype, gy.dtype),
                    RuntimeWarning)
                continue
            y.grad_var = gy
            outputs_.append(y)
        outputs = outputs_
        del outputs_

        # See also the ChainerX case of Variable.backward
        arrs = []
        for y in outputs:
            arr = y._data[0]
            assert isinstance(arr, chainerx.ndarray)
            arrs.append(arr)
        chainerx.backward(
            arrs, enable_double_backprop=enable_double_backprop)
        return

    if grad_outputs is None:
        grad_outputs = []
        for y in outputs:
            grad_var = y.grad_var
            if grad_var is None:
                warnings.warn(
                    'outputs contains a Variable without grad, or '
                    'duplicate outputs. Note that '
                    'chainer.backward does not set default grad.',
                    RuntimeWarning)
            y.grad_var = None
            grad_outputs.append(grad_var)
    outputs = [
        (y.node, gy) for y, gy in zip(outputs, grad_outputs) if gy is not None]
    with chainer.using_config('enable_backprop', enable_double_backprop):
        _backprop_to_all(outputs, False, None)


def _backprop_to_all(outputs, retain_grad, loss_scale):
    """Backprop to all input variables

    Args:
        outputs (list of tuple): each tuple is (y_node, y_grad_var).
            y_grad_var should not be None.
        retain_grad (bool): see docstring of Variable.backward
        loss_scale (float): see docstring of Variable.backward

    """
    OrderedDict = chainer.utils._collections.OrderedDict  # fix py2 memory leak

    cand_funcs = []
    seen_set = set()

    def add_cand(cand):
        if cand not in seen_set:
            # Negate since heapq is min-heap
            heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
            seen_set.add(cand)

    grads = _backprop_utils.GradTable(accumulate_grad_inputs=True)

    leaf_nodes = set()

    for y, gy in outputs:
        grads.accumulate(y, gy)

        func = y.creator_node
        if func is None:  # leaf
            leaf_nodes.add(y)
        else:
            add_cand(func)

    # Fix F812 (Python 2)
    y = None
    del y

    is_debug = chainer.is_debug()
    base_hooks = chainer.get_function_hooks().values()
    while cand_funcs:
        _, _, func = heapq.heappop(cand_funcs)
        inputs = func.inputs
        target_input_indexes = tuple([
            i for i, x in enumerate(inputs) if x.requires_grad
        ])
        outputs = [y() for y in func.outputs]  # access via weak ref
        out_grad = tuple([grads.pop(y)
                          if y is not None and y.creator_node is not None
                          else None
                          for y in outputs])
        if not target_input_indexes:
            continue

        in_data = [x.data for x in inputs]
        out_grad_array = [None if g is None else g.raw_array for g in out_grad]
        if func._n_local_function_hooks != 0:
            local_hooks = collections.OrderedDict(chainer.get_function_hooks())
            local_hooks.update(func.local_function_hooks)
            hooks = local_hooks.values()  # avoid six for performance
        else:
            hooks = base_hooks

        with chainer.using_device(
                backend.get_device_from_array(*(in_data + out_grad_array))):
            for hook in hooks:
                hook.backward_preprocess(
                    func, tuple(in_data), tuple(out_grad_array))

            # Collect the current input gradients.
            target_inputs = [inputs[i] for i in target_input_indexes]
            # Keep the order for the portability, rather than
            # in_grad = {x: grads.get_as_list(x)
            #            for x in set(target_inputs)}
            in_grad = OrderedDict()
            for x in target_inputs:
                if x not in in_grad:
                    in_grad[x] = grads.get_as_list(x)

            _backprop_utils.backprop_step(
                func, target_input_indexes, out_grad, in_grad, is_debug)

            for hook in hooks:
                hook.backward_postprocess(
                    func, tuple(in_data), tuple(out_grad_array))

        if retain_grad:
            # The gradients of the outputs of `func` are final. Store them if
            # retain_grad=True.
            for y, gy in six.moves.zip(outputs, out_grad):
                if y is not None:
                    y._set_grad_var_if_available(gy)
            del gy  # to reduce memory usage
        del out_grad  # to reduce memory usage

        for x, gx in in_grad.items():
            if not gx:  # gradient == None
                continue

            for gx_elem in gx:
                if gx_elem is not None:
                    chainer.variable._check_grad_type(
                        func, x, True, gx_elem.raw_array)
            del gx_elem  # to reduce memory usage

            if x.creator_node is None:  # leaf
                leaf_nodes.add(x)
            else:
                add_cand(x.creator_node)
        del gx, in_grad  # to reduce memory usage

    for x in leaf_nodes:
        x_var = x.get_variable_or_none()
        gx = grads.pop(x)
        if x_var is not None:
            x_var._set_grad_var_without_check(gx)
            x_var._loss_scale = loss_scale
    grads.assert_no_grads()
