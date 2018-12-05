from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import overload

from chainer import function_node
import numpy


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


@overload
def no_backprop_mode(arg0: BackpropId) -> NoBackpropMode: ...


@overload
def no_backprop_mode(arg0: List[BackpropId]) -> NoBackpropMode: ...


@overload
def no_backprop_mode(context: Optional[Context]=None) -> NoBackpropMode: ...


@overload
def force_backprop_mode(arg0: BackpropId) -> ForceBackpropMode: ...


@overload
def force_backprop_mode(arg0: List[BackpropId]) -> ForceBackpropMode: ...


@overload
def force_backprop_mode(context: Context=None) -> ForceBackpropMode: ...


@overload
def is_backprop_required(arg0: BackpropId) -> bool: ...


@overload
def is_backprop_required(context: Optional[Context]=None) -> bool: ...


# chainerx_cc/chainerx/python/backward.cc
def backward(
        arg0: Union[ndarray, List[ndarray]],
        backprop_id: Optional[BackpropId]=None,
        enable_double_backprop: bool=...) -> None: ...


# python/check_backward.cc
def check_backward(
        func: Callable[[ndarray], None],
        inputs: List[ndarray],
        grad_outputs: List[ndarray],
        eps: List[ndarray],
        atol: float=...,
        rtol: float=...,
        backprop_id: Optional[BackpropId]=None) -> None: ...


def check_double_backward(
        func: Callable[[ndarray], None],
        inputs: List[ndarray],
        grad_outputs: List[ndarray],
        grad_grad_inputs: List[ndarray],
        eps: List[ndarray],
        atol: float=...,
        rtol: float=...,
        backprop_id: Optional[BackpropId]=None) -> None: ...


# chainerx_cc/chainerx/python/context.cc
class Context:
    def get_backend(self, arg0: str) -> Backend: ...

    @overload
    def get_device(self, arg0: str) -> Device: ...

    @overload
    def get_device(self, arg0: str, arg1: int) -> Device: ...

    def make_backprop_id(self, backprop_name: str) -> BackpropId: ...

    def release_backprop_id(self, backprop_id: BackpropId) -> None: ...

    def _check_valid_backprop_id(self, backprop_id: BackpropId) -> None: ...


class ContextScope:
    def __enter__(self) -> None: ...

    def __exit__(self, *args) -> None: ...


def get_backend(arg0: str) -> Backend: ...


@overload
def get_device(device: Device) -> Device: ...


@overload
def get_device(arg0: str, arg1: Optional[int]=None) -> Device: ...


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


@overload
def set_default_device(arg0: Device) -> None: ...


@overload
def set_default_device(arg0: str) -> None: ...


@overload
def using_device(arg0: Device) -> DeviceScope: ...


@overload
def using_device(arg0: str, arg1: Optional[int]=None) -> DeviceScope: ...


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
    @property
    def dtype(self) -> numpy.dtype: ...

    @overload
    def __init__(self, value: bool) -> None: ...

    @overload
    def __init__(self, value: int) -> None: ...

    @overload
    def __init__(self, value: float) -> None: ...

    @overload
    def __init__(self, value: bool, dtype: Any) -> None: ...

    @overload
    def __init__(self, value: int, dtype: Any) -> None: ...

    @overload
    def __init__(self, value: float, dtype: Any) -> None: ...

    def __bool__(self) -> bool: ...

    def __float__(self) -> float: ...

    def __int__(self) -> int: ...

    def __neg__(self) -> Scalar: ...

    def __pos__(self) -> Scalar: ...

    def __repr__(self) -> str: ...

    def tolist(self) -> Any: ...


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
    def grad(self) -> Optional[ndarray]: ...

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
    def shape(self) -> Tuple[int, ...]: ...

    @property
    def size(self) -> int: ...

    @property
    def strides(self) -> Tuple[int, ...]: ...

    def __add__(self, arg0: Any) -> ndarray: ...

    def __bool__(self) -> bool: ...

    def __float__(self) -> float: ...

    def __ge__(self, arg0: ndarray) -> ndarray: ...

    def __getitem__(self, key: Any) -> ndarray: ...

    def __gt__(self, arg0: Any) -> ndarray: ...

    def __iadd__(self, arg0: Any) -> ndarray: ...

    def __imul__(self, arg0: Any) -> ndarray: ...

    def __init__(
            self,
            shape: Tuple[int, ...],
            dtype: Any,
            device: Optional[Device]=None) -> None: ...

    def __int__(self) -> int: ...

    def __isub__(self, arg0: Any) -> ndarray: ...

    def __itruediv__(self, arg0: Any) -> ndarray: ...

    def __le__(self, arg0: ndarray) -> ndarray: ...

    def __len__(self) -> int: ...

    def __lt__(self, arg0: Any) -> ndarray: ...

    def __mul__(self, arg0: Any) -> ndarray: ...

    def __neg__(self) -> ndarray: ...

    def __radd__(self, arg0: Any) -> ndarray: ...

    def __repr__(self) -> str: ...

    def __rmul__(self, arg0: Any) -> ndarray: ...

    def __rsub__(self, arg0: Any) -> ndarray: ...

    def __sub__(self, arg0: Any) -> ndarray: ...

    def __truediv__(self, arg0: Any) -> ndarray: ...

    def argmax(self, axis: Optional[int]=None) -> ndarray: ...

    @overload
    def as_grad_stopped(self, copy: bool=...) -> ndarray: ...

    @overload
    def as_grad_stopped(
            self,
            arg0: List[BackpropId],
            copy: bool=...) -> ndarray: ...

    def astype(self, dtype: Any, copy: bool=...) -> ndarray: ...

    def backward(
            self,
            backprop_id: Optional[BackpropId]=None,
            enable_double_backprop: bool=...) -> None: ...

    def cleargrad(self, backprop_id: Optional[BackpropId]=None) -> None: ...

    def clip(self, a_min: Optional[int], a_max: Optional[int]): ...

    def copy(self) -> ndarray: ...

    def dot(self, b: ndarray) -> ndarray: ...

    def fill(self, value: Any) -> None: ...

    def get_grad(self, backprop_id: Optional[BackpropId]=None) -> ndarray: ...

    @overload
    def is_backprop_required(
            self,
            backprop_id:
            Optional[BackpropId]=None) -> bool: ...

    @overload
    def is_backprop_required(self, backprop_id: AnyGraph) -> bool: ...

    def is_grad_required(
            self,
            backprop_id: Optional[BackpropId]=None) -> bool: ...

    @overload
    def max(self,
            axis: int,
            keepdims: bool=...) -> ndarray: ...

    @overload
    def max(self,
            axis: Optional[Tuple[int, ...]]=None,
            keepdims: bool=...) -> ndarray: ...

    def ravel(self) -> ndarray: ...

    def require_grad(
            self,
            backprop_id: Optional[BackpropId]=None) -> ndarray: ...

    @overload
    def reshape(self, arg0: Tuple[int, ...]) -> ndarray: ...

    @overload
    def reshape(self, arg0: List[int]) -> ndarray: ...

    @overload
    def reshape(self, *args: Any) -> ndarray: ...

    def set_grad(
            self,
            grad: ndarray,
            backprop_id: Optional[BackpropId]=None) -> None: ...

    @overload
    def squeeze(self, axis: Optional[Tuple[int, ...]]=None) -> ndarray: ...

    @overload
    def squeeze(self, axis: int) -> ndarray: ...

    @overload
    def sum(self, axis: int, keepdims: bool=...) -> ndarray: ...

    @overload
    def sum(self,
            axis: Optional[Tuple[int, ...]]=None,
            keepdims: bool=...) -> ndarray: ...

    def take(self, indices: ndarray, axis: Optional[int]=None) -> ndarray: ...

    @overload
    def to_device(self, arg0: Device) -> ndarray: ...

    @overload
    def to_device(self, arg0: str, arg1: int) -> ndarray: ...

    @overload
    def transpose(
            self,
            axes: Optional[List[int]]=None) -> ndarray: ...

    @overload
    def transpose(self, *args: Any) -> ndarray: ...

    @overload
    def transpose(self, axes: int) -> ndarray: ...

    def view(self) -> ndarray: ...


# chainerx_cc/chainerx/python/routines.cc
def add(x1: Any, x2: Any) -> ndarray: ...


def amax(a: ndarray,
         axis: Union[int, Optional[List[int]]]=None,
         keepdims: bool=...) -> ndarray: ...


def arange(
        start: Any,
        stop: Optional[Any]=None,
        step: Optional[Any]=None,
        dtype: Optional[Any]=None,
        device: Optional[Device]=None) -> ndarray: ...


def argmax(a: ndarray, axis: Optional[int]=None) -> ndarray: ...


def array(
        object: Any,
        dtype: Optional[Any]=None,
        copy: bool=...,
        device: Optional[Device]=None) -> ndarray: ...


def asarray(
        object: Any,
        dtype: Optional[Any]=None,
        device: Optional[Device]=None) -> ndarray: ...


def ascontiguousarray(
        a: Any,
        dtype: Optional[Any]=None,
        device: Optional[Device]=None) -> ndarray: ...


def asscalar(a: ndarray) -> Any: ...


def average_pool(
        x: ndarray,
        ksize: Union[int, Tuple[int, ...]],
        stride: Optional[Union[int, Tuple[int, ...]]]=None,
        pad: Union[int, Tuple[int, ...]]=...,
        pad_mode: str=...) -> ndarray: ...


def backprop_scope(backprop_name: str, context: Any=None) -> BackpropScope: ...


def batch_norm(
        x: ndarray,
        gamma: ndarray,
        beta: ndarray,
        running_mean: ndarray,
        running_var: ndarray,
        eps: float=...,
        decay: float=...,
        axis: Optional[Union[int, Tuple[int, ...]]]=None) -> ndarray: ...


def broadcast_to(array: ndarray, shape: Tuple[int, ...]) -> ndarray: ...


def concatenate(arrays: List[ndarray], axis: Optional[int]=...) -> ndarray: ...


def context_scope(arg0: Context) -> ContextScope: ...


def conv(
        x: ndarray,
        w: ndarray,
        b: Optional[ndarray]=None,
        stride: Union[int, Tuple[int, ...]]=...,
        pad: Union[int, Tuple[int, ...]]=...,
        cover_all: bool=False) -> ndarray: ...


def conv_transpose(
        x: ndarray,
        w: ndarray,
        b: Optional[ndarray]=None,
        stride: Union[int, Tuple[int, ...]]=...,
        pad: Union[int, Tuple[int, ...]]=...,
        outsize: Optional[Tuple[int, ...]]=None) -> ndarray: ...


def copy(a: ndarray) -> ndarray: ...


def diag(v: ndarray, k: int=..., device: Optional[Device]=None) -> ndarray: ...


def diagflat(
        v: ndarray,
        k: int=...,
        device: Optional[Device]=None) -> ndarray: ...


def divide(x1: Any, x2: Any) -> ndarray: ...


def dot(a: ndarray, b: ndarray) -> ndarray: ...


def empty(
        shape: Union[int, Tuple[int, ...]],
        dtype: Optional[Any]=None,
        device: Optional[Device]=None) -> ndarray: ...


def empty_like(a: ndarray, device: Optional[Device]=None) -> ndarray: ...


def equal(x1: ndarray, x2: ndarray) -> ndarray: ...


def exp(x: ndarray) -> ndarray: ...


def eye(N: int,
        M: Optional[int]=None,
        k: int=...,
        dtype: Optional[Any]=...,
        device: Optional[Device]=None) -> ndarray: ...


def fixed_batch_norm(
        x: ndarray,
        gamma: ndarray,
        beta: ndarray,
        mean: ndarray,
        var: ndarray,
        eps: float=...,
        axis: Optional[Union[int, List[int]]]=None) -> ndarray: ...


def frombuffer(
        buffer: Any,
        dtype: Optional[Any]=...,
        count: int=...,
        offset: int=...,
        device: Optional[Device]=None) -> ndarray: ...


def full(
        shape: Union[int, Tuple[int, ...]],
        fill_value: Any,
        dtype: Optional[Any],
        device: Optional[Device]=None) -> ndarray: ...


def full_like(
        a: ndarray,
        fill_value: Any,
        device: Optional[Device]=None) -> ndarray: ...


def greater(x1: ndarray, x2: ndarray) -> ndarray: ...


def greater_equal(x1: ndarray, x2: ndarray) -> ndarray: ...


def identity(
        n: int,
        dtype: Optional[Any]=None,
        device: Optional[Device]=None) -> ndarray: ...


def is_available(): ...


def isinf(x: ndarray) -> ndarray: ...


def isnan(x: ndarray) -> ndarray: ...


def less(x1: ndarray, x2: ndarray) -> ndarray: ...


def less_equal(x1: ndarray, x2: ndarray) -> ndarray: ...


def linear(
        x: ndarray,
        w: ndarray,
        b: Optional[ndarray]=None,
        n_batch_axes: int=1) -> ndarray: ...


def linspace(
        start: Any,
        stop: Any,
        num: int=...,
        endpoint: bool=True,
        dtype: Optional[Any]=None,
        device: Optional[Device]=None) -> ndarray: ...


def log(x: ndarray) -> ndarray: ...


def log_softmax(
        x: ndarray,
        axis: Optional[Union[int, List[int]]]=None) -> ndarray: ...


def logical_not(x: ndarray) -> ndarray: ...


def logsumexp(
        x: ndarray,
        axis: Optional[Union[int, List[int]]]=None,
        keepdims: bool=...) -> ndarray: ...


def max_pool(
        x: ndarray,
        ksize: Any,
        stride: Union[int, Tuple[int, ...]]=None,
        pad: Union[int, Tuple[int, ...]]=...,
        cover_all: bool=...) -> ndarray: ...


def maximum(x1: Any, x2: Any) -> ndarray: ...


def multiply(x1: Any, x2: Any) -> ndarray: ...


def negative(x: ndarray) -> ndarray: ...


def not_equal(x1: ndarray, x2: ndarray) -> ndarray: ...


def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[Any]=None,
         device: Optional[Device]=None) -> ndarray: ...


def ones_like(a: ndarray, device: Optional[Device]=None) -> ndarray: ...


@overload
def reshape(
        a: ndarray,
        newshape: Union[int, Tuple[int, ...]]) -> ndarray: ...


@overload
def reshape(
        a: ndarray,
        newshape: Union[int, List[int]]) -> ndarray: ...


@overload
def reshape(a: ndarray, *args: Any) -> ndarray: ...


def split(
        ary: ndarray,
        indices_or_sections: Union[int, List[int]],
        axis: int=...) -> List[ndarray]: ...


def sqrt(x: ndarray) -> ndarray: ...


def squeeze(
        a: ndarray,
        axis: Optional[Union[int, List[int]]]=None) -> ndarray: ...


def stack(arrays: List[ndarray], axis: int=...) -> ndarray: ...


def subtract(x1: Any, x2: Any) -> ndarray: ...


def sum(a: ndarray,
        axis: Optional[Union[int, List[int]]]=None,
        keepdims: bool=...) -> ndarray: ...


def take(a: ndarray, indices: ndarray, axis: Optional[int]) -> ndarray: ...


def tanh(x: ndarray) -> ndarray: ...


def to_numpy(array: ndarray, copy: bool=...) -> numpy.ndarray: ...


def transpose(
        a: ndarray,
        axes: Optional[Union[int, List[int]]]=None) -> ndarray: ...


def zeros(
        shape: Union[int, Tuple[int, ...]],
        dtype: Optional[Any]=None,
        device: Optional[Device]=None) -> ndarray: ...


def zeros_like(a: ndarray, device: Optional[Device]=None) -> ndarray: ...


# chainerx_cc/chainerx/python/chainer_interop.cc
def _function_node_forward(
        function_node: function_node.FunctionNode,
        inputs: List[ndarray],
        outputs: List[ndarray],
        input_indexes_to_retain: List[int],
        output_indexes_to_retain: List[int]) -> None: ...


# chainerx/creation/from_data.py
def asanyarray(
        a: ndarray,
        dtype: Optional[Any]=None,
        device: Optional[Device]=None) -> ndarray: ...


def fromfile(
        file: str,
        dtype: Optional[Any]=...,
        count: int=...,
        sep: str=...,
        device: Optional[Device]=None) -> ndarray: ...


def fromfunction(
        function: Callable[..., Any],
        shape: Tuple[int, ...],
        **kwargs: Any) -> ndarray: ...


def fromiter(
        iterable: Iterable[Any],
        dtype: Optional[Any],
        count: int=...,
        device: Optional[Device]=None) -> ndarray: ...


def fromstring(
        string: str,
        dtype: Optional[Any]=float,
        count=...,
        sep=...,
        device=None) -> ndarray: ...


def loadtxt(
        fname: str,
        dtype: Optional[Any]=...,
        comments: Optional[Union[str, Iterable[str]]]=...,
        delimiter: Optional[str]=None,
        converters: Optional[Dict[int, Callable[[str], Any]]]=None,
        skiprows: int=...,
        usecols: Optional[Union[int, Iterable[int]]]=None,
        unpack: bool=...,
        ndmin: int=...,
        encoding: Optional[str]=...,
        device: Optional[Device]=None) -> ndarray: ...


# chainerx/activation.py
def relu(x: ndarray) -> ndarray: ...


def sigmoid(x: ndarray) -> ndarray: ...


# chainerx/manipulation/shape.py
def ravel(a: ndarray) -> ndarray: ...


# chainerx/math/misc.py
def square(x: ndarray) -> ndarray: ...


def clip(a: ndarray, a_min: Any, a_max: Any) -> ndarray: ...
