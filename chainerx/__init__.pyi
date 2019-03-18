import typing as tp

import numpy

from chainer import function_node


# TODO(okapies): Split this into independent .py and .pyi files
# mypy can't handle wildcard imports properly, so I aggregate
# the signatures under chainerx module to this place.
# See https://github.com/python/mypy/issues/5908 for the details


# chainerx_cc/chainerx/python/backend.cc
class Backend:
    @property
    def context(self) -> Context: ...

    @property
    def name(self) -> str: ...

    def get_device(self, arg0: int) -> Device: ...

    def get_device_count(self) -> int: ...


# chainerx_cc/chainerx/python/backprop_mode.cc
class NoBackpropMode:
    def __enter__(self) -> None: ...

    def __exit__(self, *args) -> None: ...


class ForceBackpropMode:
    def __enter__(self) -> None: ...

    def __exit__(self, *args) -> None: ...


@tp.overload
def no_backprop_mode(arg0: BackpropId) -> NoBackpropMode: ...


@tp.overload
def no_backprop_mode(arg0: tp.List[BackpropId]) -> NoBackpropMode: ...


@tp.overload
def no_backprop_mode(context: tp.Optional[Context]=None) -> NoBackpropMode: ...


@tp.overload
def force_backprop_mode(arg0: BackpropId) -> ForceBackpropMode: ...


@tp.overload
def force_backprop_mode(arg0: tp.List[BackpropId]) -> ForceBackpropMode: ...


@tp.overload
def force_backprop_mode(context: Context=None) -> ForceBackpropMode: ...


@tp.overload
def is_backprop_required(arg0: BackpropId) -> bool: ...


@tp.overload
def is_backprop_required(context: tp.Optional[Context]=None) -> bool: ...


# chainerx_cc/chainerx/python/backward.cc
def backward(
        arg0: tp.Union[ndarray, tp.List[ndarray]],
        backprop_id: tp.Optional[BackpropId]=None,
        enable_double_backprop: bool=...) -> None: ...


# python/check_backward.cc
def check_backward(
        func: tp.Callable[[ndarray], None],
        inputs: tp.List[ndarray],
        grad_outputs: tp.List[ndarray],
        eps: tp.List[ndarray],
        atol: float=...,
        rtol: float=...,
        backprop_id: tp.Optional[BackpropId]=None) -> None: ...


def check_double_backward(
        func: tp.Callable[[ndarray], None],
        inputs: tp.List[ndarray],
        grad_outputs: tp.List[ndarray],
        grad_grad_inputs: tp.List[ndarray],
        eps: tp.List[ndarray],
        atol: float=...,
        rtol: float=...,
        backprop_id: tp.Optional[BackpropId]=None) -> None: ...


# chainerx_cc/chainerx/python/context.cc
class Context:
    def get_backend(self, arg0: str) -> Backend: ...

    @tp.overload
    def get_device(self, arg0: str) -> Device: ...

    @tp.overload
    def get_device(self, arg0: str, arg1: int) -> Device: ...

    def make_backprop_id(self, backprop_name: str) -> BackpropId: ...

    def release_backprop_id(self, backprop_id: BackpropId) -> None: ...

    def _check_valid_backprop_id(self, backprop_id: BackpropId) -> None: ...


class ContextScope:
    def __enter__(self) -> None: ...

    def __exit__(self, *args) -> None: ...


def get_backend(arg0: str) -> Backend: ...


@tp.overload
def get_device(device: Device) -> Device: ...


@tp.overload
def get_device(arg0: str, arg1: tp.Optional[int]=None) -> Device: ...


def get_default_context() -> Context: ...


def set_default_context(arg0: Context) -> None: ...


def get_global_default_context() -> Context: ...


def set_global_default_context(arg0: Context) -> None: ...


# chainerx_cc/chainerx/python/device.cc
class Device:
    @property
    def backend(self) -> Backend: ...

    @property
    def context(self) -> Context: ...

    @property
    def index(self) -> int: ...

    @property
    def name(self) -> str: ...

    def synchronize(self) -> None: ...


def get_default_device() -> Device: ...


@tp.overload
def set_default_device(arg0: Device) -> None: ...


@tp.overload
def set_default_device(arg0: str) -> None: ...


@tp.overload
def using_device(arg0: Device) -> DeviceScope: ...


@tp.overload
def using_device(arg0: str, arg1: tp.Optional[int]=None) -> DeviceScope: ...


class DeviceScope:
    def __enter__(self) -> DeviceScope: ...

    def __exit__(self, *args) -> None: ...

    @property
    def device(self) -> Device: ...


# chainerx_cc/chainerx/python/error.cc
class BackendError(Exception): ...


class ChainerxError(Exception): ...


class ContextError(Exception): ...


class DeviceError(Exception): ...


class DimensionError(Exception): ...


class DtypeError(Exception): ...


class GradientCheckError(Exception): ...


class NotImplementedError(Exception): ...


# chainerx_cc/chainerx/python/graph.cc
class AnyGraph: ...


class BackpropId:
    @property
    def context(self) -> Context: ...

    @property
    def name(self) -> str: ...


class BackpropScope:
    def __enter__(self) -> BackpropId: ...

    def __exit__(self, *args) -> None: ...


# chainerx_cc/chainerx/python/scalar.cc
class Scalar:
    @tp.overload
    def __init__(self, value: bool) -> None: ...

    @tp.overload
    def __init__(self, value: int) -> None: ...

    @tp.overload
    def __init__(self, value: float) -> None: ...

    @tp.overload
    def __init__(self, value: bool, dtype: tp.Any) -> None: ...

    @tp.overload
    def __init__(self, value: int, dtype: tp.Any) -> None: ...

    @tp.overload
    def __init__(self, value: float, dtype: tp.Any) -> None: ...

    def __bool__(self) -> bool: ...

    def __float__(self) -> float: ...

    def __int__(self) -> int: ...

    def __neg__(self) -> Scalar: ...

    def __pos__(self) -> Scalar: ...

    def __repr__(self) -> str: ...

    def toList(self) -> tp.Any: ...


# chainerx_cc/chainerx/python/array.cc
class ndarray:
    @property
    def T(self) -> ndarray: ...

    @property
    def data_ptr(self) -> int: ...

    @property
    def data_size(self) -> int: ...

    @property
    def device(self) -> Device: ...

    @property
    def dtype(self) -> numpy.dtype: ...

    @property
    def grad(self) -> tp.Optional[ndarray]: ...

    @property
    def is_contiguous(self) -> bool: ...

    @property
    def itemsize(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    @property
    def ndim(self) -> int: ...

    @property
    def offset(self) -> int: ...

    @property
    def shape(self) -> tp.Tuple[int, ...]: ...

    @property
    def size(self) -> int: ...

    @property
    def strides(self) -> tp.Tuple[int, ...]: ...

    def __add__(self, arg0: tp.Any) -> ndarray: ...

    def __bool__(self) -> bool: ...

    def __float__(self) -> float: ...

    def __ge__(self, arg0: ndarray) -> ndarray: ...

    def __getitem__(self, key: tp.Any) -> ndarray: ...

    def __gt__(self, arg0: tp.Any) -> ndarray: ...

    def __iadd__(self, arg0: tp.Any) -> ndarray: ...

    def __imul__(self, arg0: tp.Any) -> ndarray: ...

    def __init__(
            self,
            shape: tp.Tuple[int, ...],
            dtype: tp.Any,
            device: tp.Optional[Device]=None) -> None: ...

    def __int__(self) -> int: ...

    def __isub__(self, arg0: tp.Any) -> ndarray: ...

    def __itruediv__(self, arg0: tp.Any) -> ndarray: ...

    def __le__(self, arg0: ndarray) -> ndarray: ...

    def __len__(self) -> int: ...

    def __lt__(self, arg0: tp.Any) -> ndarray: ...

    def __mul__(self, arg0: tp.Any) -> ndarray: ...

    def __neg__(self) -> ndarray: ...

    def __radd__(self, arg0: tp.Any) -> ndarray: ...

    def __repr__(self) -> str: ...

    def __rmul__(self, arg0: tp.Any) -> ndarray: ...

    def __rsub__(self, arg0: tp.Any) -> ndarray: ...

    def __sub__(self, arg0: tp.Any) -> ndarray: ...

    def __truediv__(self, arg0: tp.Any) -> ndarray: ...

    def argmax(self, axis: tp.Optional[int]=None) -> ndarray: ...

    @tp.overload
    def as_grad_stopped(self, copy: bool=...) -> ndarray: ...

    @tp.overload
    def as_grad_stopped(
            self,
            arg0: tp.List[BackpropId],
            copy: bool=...) -> ndarray: ...

    def astype(self, dtype: tp.Any, copy: bool=...) -> ndarray: ...

    def backward(
            self,
            backprop_id: tp.Optional[BackpropId]=None,
            enable_double_backprop: bool=...) -> None: ...

    def cleargrad(self, backprop_id: tp.Optional[BackpropId]=None) -> None: ...

    def clip(self, a_min: tp.Optional[int], a_max: tp.Optional[int]): ...

    def copy(self) -> ndarray: ...

    def dot(self, b: ndarray) -> ndarray: ...

    def fill(self, value: tp.Any) -> None: ...

    def get_grad(self, backprop_id: tp.Optional[BackpropId]=None) -> ndarray: ...

    @tp.overload
    def is_backprop_required(
            self,
            backprop_id:
            tp.Optional[BackpropId]=None) -> bool: ...

    @tp.overload
    def is_backprop_required(self, backprop_id: AnyGraph) -> bool: ...

    def is_grad_required(
            self,
            backprop_id: tp.Optional[BackpropId]=None) -> bool: ...

    @tp.overload
    def max(self,
            axis: int,
            keepdims: bool=...) -> ndarray: ...

    @tp.overload
    def max(self,
            axis: tp.Optional[tp.Tuple[int, ...]]=None,
            keepdims: bool=...) -> ndarray: ...

    def ravel(self) -> ndarray: ...

    def require_grad(
            self,
            backprop_id: tp.Optional[BackpropId]=None) -> ndarray: ...

    @tp.overload
    def reshape(self, arg0: tp.Tuple[int, ...]) -> ndarray: ...

    @tp.overload
    def reshape(self, arg0: tp.List[int]) -> ndarray: ...

    @tp.overload
    def reshape(self, *args: tp.Any) -> ndarray: ...

    def set_grad(
            self,
            grad: ndarray,
            backprop_id: tp.Optional[BackpropId]=None) -> None: ...

    @tp.overload
    def squeeze(self, axis: tp.Optional[tp.Tuple[int, ...]]=None) -> ndarray: ...

    @tp.overload
    def squeeze(self, axis: int) -> ndarray: ...

    @tp.overload
    def sum(self, axis: int, keepdims: bool=...) -> ndarray: ...

    @tp.overload
    def sum(self,
            axis: tp.Optional[tp.Tuple[int, ...]]=None,
            keepdims: bool=...) -> ndarray: ...

    def take(self, indices: tp.Union[tp.Sequence[int], numpy.ndarray, ndarray],
             axis: tp.Optional[int]=None) -> ndarray: ...

    @tp.overload
    def to_device(self, arg0: Device) -> ndarray: ...

    @tp.overload
    def to_device(self, arg0: str, arg1: int) -> ndarray: ...

    @tp.overload
    def transpose(
            self,
            axes: tp.Optional[tp.List[int]]=None) -> ndarray: ...

    @tp.overload
    def transpose(self, *args: tp.Any) -> ndarray: ...

    @tp.overload
    def transpose(self, axes: int) -> ndarray: ...

    def view(self) -> ndarray: ...


# chainerx_cc/chainerx/python/routines.cc
def add(x1: tp.Any, x2: tp.Any) -> ndarray: ...


def amax(a: ndarray,
         axis: tp.Union[int, tp.Optional[tp.List[int]]]=None,
         keepdims: bool=...) -> ndarray: ...


def arange(
        start: tp.Any,
        stop: tp.Optional[tp.Any]=None,
        step: tp.Optional[tp.Any]=None,
        dtype: tp.Optional[tp.Any]=None,
        device: tp.Optional[Device]=None) -> ndarray: ...


def argmax(a: ndarray, axis: tp.Optional[int]=None) -> ndarray: ...


def array(
        object: tp.Any,
        dtype: tp.Optional[tp.Any]=None,
        copy: bool=...,
        device: tp.Optional[Device]=None) -> ndarray: ...


def asarray(
        object: tp.Any,
        dtype: tp.Optional[tp.Any]=None,
        device: tp.Optional[Device]=None) -> ndarray: ...


def ascontiguousarray(
        a: tp.Any,
        dtype: tp.Optional[tp.Any]=None,
        device: tp.Optional[Device]=None) -> ndarray: ...


def average_pool(
        x: ndarray,
        ksize: tp.Union[int, tp.Tuple[int, ...]],
        stride: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]]=None,
        pad: tp.Union[int, tp.Tuple[int, ...]]=...,
        pad_mode: str=...) -> ndarray: ...


def backprop_scope(backprop_name: str, context: tp.Any=None) -> BackpropScope: ...


def batch_norm(
        x: ndarray,
        gamma: ndarray,
        beta: ndarray,
        running_mean: ndarray,
        running_var: ndarray,
        eps: float=...,
        decay: float=...,
        axis: tp.Optional[tp.Union[int, tp.Tuple[int, ...]]]=None) -> ndarray: ...


def broadcast_to(array: ndarray, shape: tp.Tuple[int, ...]) -> ndarray: ...


def concatenate(arrays: tp.List[ndarray], axis: tp.Optional[int]=...) -> ndarray: ...


def context_scope(arg0: Context) -> ContextScope: ...


def conv(
        x: ndarray,
        w: ndarray,
        b: tp.Optional[ndarray]=None,
        stride: tp.Union[int, tp.Tuple[int, ...]]=...,
        pad: tp.Union[int, tp.Tuple[int, ...]]=...,
        cover_all: bool=False) -> ndarray: ...


def conv_transpose(
        x: ndarray,
        w: ndarray,
        b: tp.Optional[ndarray]=None,
        stride: tp.Union[int, tp.Tuple[int, ...]]=...,
        pad: tp.Union[int, tp.Tuple[int, ...]]=...,
        outsize: tp.Optional[tp.Tuple[int, ...]]=None) -> ndarray: ...


def copy(a: ndarray) -> ndarray: ...


def diag(v: ndarray, k: int=..., device: tp.Optional[Device]=None) -> ndarray: ...


def diagflat(
        v: ndarray,
        k: int=...,
        device: tp.Optional[Device]=None) -> ndarray: ...


def divide(x1: tp.Any, x2: tp.Any) -> ndarray: ...


def dot(a: ndarray, b: ndarray) -> ndarray: ...


def empty(
        shape: tp.Union[int, tp.Tuple[int, ...]],
        dtype: tp.Optional[tp.Any]=None,
        device: tp.Optional[Device]=None) -> ndarray: ...


def empty_like(a: ndarray, device: tp.Optional[Device]=None) -> ndarray: ...


def equal(x1: ndarray, x2: ndarray) -> ndarray: ...


def exp(x: ndarray) -> ndarray: ...


def eye(N: int,
        M: tp.Optional[int]=None,
        k: int=...,
        dtype: tp.Optional[tp.Any]=...,
        device: tp.Optional[Device]=None) -> ndarray: ...


def fixed_batch_norm(
        x: ndarray,
        gamma: ndarray,
        beta: ndarray,
        mean: ndarray,
        var: ndarray,
        eps: float=...,
        axis: tp.Optional[tp.Union[int, tp.List[int]]]=None) -> ndarray: ...


def frombuffer(
        buffer: tp.Any,
        dtype: tp.Optional[tp.Any]=...,
        count: int=...,
        offset: int=...,
        device: tp.Optional[Device]=None) -> ndarray: ...


def full(
        shape: tp.Union[int, tp.Tuple[int, ...]],
        fill_value: tp.Any,
        dtype: tp.Optional[tp.Any],
        device: tp.Optional[Device]=None) -> ndarray: ...


def full_like(
        a: ndarray,
        fill_value: tp.Any,
        device: tp.Optional[Device]=None) -> ndarray: ...


def greater(x1: ndarray, x2: ndarray) -> ndarray: ...


def greater_equal(x1: ndarray, x2: ndarray) -> ndarray: ...


def identity(
        n: int,
        dtype: tp.Optional[tp.Any]=None,
        device: tp.Optional[Device]=None) -> ndarray: ...


def is_available(): ...


def isinf(x: ndarray) -> ndarray: ...


def isnan(x: ndarray) -> ndarray: ...


def less(x1: ndarray, x2: ndarray) -> ndarray: ...


def less_equal(x1: ndarray, x2: ndarray) -> ndarray: ...


def linear(
        x: ndarray,
        w: ndarray,
        b: tp.Optional[ndarray]=None,
        n_batch_axes: int=1) -> ndarray: ...


def linspace(
        start: tp.Any,
        stop: tp.Any,
        num: int=...,
        endpoint: bool=True,
        dtype: tp.Optional[tp.Any]=None,
        device: tp.Optional[Device]=None) -> ndarray: ...


def log(x: ndarray) -> ndarray: ...


def log_softmax(
        x: ndarray,
        axis: tp.Optional[tp.Union[int, tp.List[int]]]=None) -> ndarray: ...


def logical_not(x: ndarray) -> ndarray: ...


def logsumexp(
        x: ndarray,
        axis: tp.Optional[tp.Union[int, tp.List[int]]]=None,
        keepdims: bool=...) -> ndarray: ...


def max_pool(
        x: ndarray,
        ksize: tp.Any,
        stride: tp.Union[int, tp.Tuple[int, ...]]=None,
        pad: tp.Union[int, tp.Tuple[int, ...]]=...,
        cover_all: bool=...) -> ndarray: ...


def maximum(x1: tp.Any, x2: tp.Any) -> ndarray: ...


def minimum(x1: tp.Any, x2: tp.Any) -> ndarray: ...


def multiply(x1: tp.Any, x2: tp.Any) -> ndarray: ...


def negative(x: ndarray) -> ndarray: ...


def not_equal(x1: ndarray, x2: ndarray) -> ndarray: ...


def ones(shape: tp.Union[int, tp.Tuple[int, ...]],
         dtype: tp.Optional[tp.Any]=None,
         device: tp.Optional[Device]=None) -> ndarray: ...


def ones_like(a: ndarray, device: tp.Optional[Device]=None) -> ndarray: ...


@tp.overload
def reshape(
        a: ndarray,
        newshape: tp.Union[int, tp.Tuple[int, ...]]) -> ndarray: ...


@tp.overload
def reshape(
        a: ndarray,
        newshape: tp.Union[int, tp.List[int]]) -> ndarray: ...


@tp.overload
def reshape(a: ndarray, *args: tp.Any) -> ndarray: ...


def split(
        ary: ndarray,
        indices_or_sections: tp.Union[int, tp.List[int]],
        axis: int=...) -> tp.List[ndarray]: ...


def sqrt(x: ndarray) -> ndarray: ...


def squeeze(
        a: ndarray,
        axis: tp.Optional[tp.Union[int, tp.List[int]]]=None) -> ndarray: ...


def stack(arrays: tp.List[ndarray], axis: int=...) -> ndarray: ...


def subtract(x1: tp.Any, x2: tp.Any) -> ndarray: ...


def sum(a: ndarray,
        axis: tp.Optional[tp.Union[int, tp.List[int]]]=None,
        keepdims: bool=...) -> ndarray: ...


def take(a: ndarray, indices: ndarray, axis: tp.Optional[int]) -> ndarray: ...


def tanh(x: ndarray) -> ndarray: ...


def to_numpy(array: ndarray, copy: bool=...) -> numpy.ndarray: ...


def transpose(
        a: ndarray,
        axes: tp.Optional[tp.Union[int, tp.List[int]]]=None) -> ndarray: ...


def zeros(
        shape: tp.Union[int, tp.Tuple[int, ...]],
        dtype: tp.Optional[tp.Any]=None,
        device: tp.Optional[Device]=None) -> ndarray: ...


def zeros_like(a: ndarray, device: tp.Optional[Device]=None) -> ndarray: ...


# chainerx_cc/chainerx/python/chainer_interop.cc
def _function_node_forward(
        function_node: function_node.FunctionNode,
        inputs: tp.List[ndarray],
        outputs: tp.List[ndarray],
        input_indexes_to_retain: tp.List[int],
        output_indexes_to_retain: tp.List[int]) -> None: ...


# chainerx/creation/from_data.py
def asanyarray(
        a: ndarray,
        dtype: tp.Optional[tp.Any]=None,
        device: tp.Optional[Device]=None) -> ndarray: ...


def fromfile(
        file: str,
        dtype: tp.Optional[tp.Any]=...,
        count: int=...,
        sep: str=...,
        device: tp.Optional[Device]=None) -> ndarray: ...


def fromfunction(
        function: tp.Callable[..., tp.Any],
        shape: tp.Tuple[int, ...],
        **kwargs: tp.Any) -> ndarray: ...


def fromiter(
        iterable: tp.Iterable[tp.Any],
        dtype: tp.Optional[tp.Any],
        count: int=...,
        device: tp.Optional[Device]=None) -> ndarray: ...


def fromstring(
        string: str,
        dtype: tp.Optional[tp.Any]=float,
        count=...,
        sep=...,
        device=None) -> ndarray: ...


def loadtxt(
        fname: str,
        dtype: tp.Optional[tp.Any]=...,
        comments: tp.Optional[tp.Union[str, tp.Iterable[str]]]=...,
        delimiter: tp.Optional[str]=None,
        converters: tp.Optional[tp.Dict[int, tp.Callable[[str], tp.Any]]]=None,
        skiprows: int=...,
        usecols: tp.Optional[tp.Union[int, tp.Iterable[int]]]=None,
        unpack: bool=...,
        ndmin: int=...,
        encoding: tp.Optional[str]=...,
        device: tp.Optional[Device]=None) -> ndarray: ...


# chainerx/activation.py
def relu(x: ndarray) -> ndarray: ...


def sigmoid(x: ndarray) -> ndarray: ...


# chainerx/manipulation/shape.py
def ravel(a: ndarray) -> ndarray: ...


# chainerx/math/misc.py
def square(x: ndarray) -> ndarray: ...


def clip(a: ndarray, a_min: tp.Any, a_max: tp.Any) -> ndarray: ...
