import numpy as np

from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer

_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.lr = 0.01
_default_hyperparam.momentum = 0.9


class MomentumSGDRule(optimizer.UpdateRule):

    """Update rule for the classical momentum SGD.

    See :class:`~chainer.optimizers.MomentumSGD` for the default values of the
    hyperparameters.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """
    _kernel = None

    def __init__(self, parent_hyperparam=None, lr=None, momentum=None):
        super(MomentumSGDRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if momentum is not None:
            self.hyperparam.momentum = momentum

    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['v'] = xp.zeros_like(param.data)

        # For iDeep
        if (isinstance(param.data, intel64.mdarray) and
                intel64.inputs_all_ready((self.state['v'],))):
            self.state['v'] = intel64.ideep.array(
                self.state['v'], itype=intel64.ideep.wgt_array)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        v = self.state['v']
        if isinstance(v, intel64.mdarray):
            v.inplace_axpby(self.hyperparam.momentum, -
                            self.hyperparam.lr, grad)
            param.data += v
        else:
            v *= self.hyperparam.momentum
            v -= self.hyperparam.lr * grad
            param.data += v

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        if MomentumSGDRule._kernel is None:
            MomentumSGDRule._kernel = cuda.elementwise(
                'T grad, T lr, T momentum',
                'T param, T v',
                '''v = momentum * v - lr * grad;
                   param += v;''',
                'momentum_sgd')
        MomentumSGDRule._kernel(
            grad, self.hyperparam.lr, self.hyperparam.momentum, param.data,
            self.state['v'])

    def batched_update(self, params, loss_scale=None):
        if loss_scale is None:
            loss_scale = 1.0
        lr = cuda.cupy.array(self.hyperparam.lr / loss_scale, dtype=np.float32)
        momentum = cuda.cupy.array(self.hyperparam.momentum, dtype=np.float32)
        pinfo = ParamsInfo(params)
        n_threads = 128
        n_blocks = (pinfo.n_elems + n_threads - 1) // n_threads
        _cupy_batched_momentum_sgd()(
            (n_blocks, ), (n_threads, ),
            (pinfo.data_ptr, pinfo.data_dtype, pinfo.grad_ptr, pinfo.v_ptr,
             pinfo.fp32_data_ptr, pinfo.size_csum, pinfo.n_params,
             pinfo.n_elems, lr, momentum))


def _cupy_batched_momentum_sgd():
    return cuda.cupy.RawKernel(r'''
#include <cuda_fp16.h>
#define FLOAT16  6
#define FLOAT32  7
    extern "C" __global__
    void cupy_batched_momentum_sgd(
            unsigned long *data_ptr, const int *data_dtype,
            const unsigned long *grad_ptr,
            unsigned long *v_ptr,
            unsigned long *fp32_data_ptr,
            const int *size_csum, int n_params, int n_elems,
            const float *lr, const float *momentum)
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= n_elems) return;
        int j_min = 0;
        int j_max = n_params - 1;
        int j;
        while (1) {
            j = (j_min + j_max) / 2;
            if (tid < size_csum[j]) {
                j_max = j - 1;
                continue;
            }
            if (tid >= size_csum[j+1]){
                j_min = j + 1;
                continue;
            }
            break;
        }
        int idx = tid - size_csum[j];
        if (fp32_data_ptr[j] != 0) {
            // fp32 update
            float *fp32_data = (float*)(fp32_data_ptr[j]) + idx;
            float *v = (float*)(v_ptr[j]) + idx;
            half *data = (half*)(data_ptr[j]) + idx;
            half *grad = (half*)(grad_ptr[j]) + idx;
            v[0] = v[0] * momentum[0] - (float)grad[0] * lr[0];
            fp32_data[0] += v[0];
            data[0] = (half)fp32_data[0];
        }
        else if (data_dtype[j] == FLOAT16) {
            half *data = (half*)(data_ptr[j]) + idx;
            half *v = (half*)(v_ptr[j]) + idx;
            half *grad = (half*)(grad_ptr[j]) + idx;
            v[0] = v[0] * (half)momentum[0] - grad[0] * (half)lr[0];
            data[0] += v[0];
        }
        else if (data_dtype[j] == FLOAT32) {
            float *data = (float*)(data_ptr[j]) + idx;
            float *v = (float*)(v_ptr[j]) + idx;
            float *grad = (float*)(grad_ptr[j]) + idx;
            v[0] = v[0] * momentum[0] - grad[0] * lr[0];
            data[0] += v[0];
        }
    }

    ''', 'cupy_batched_momentum_sgd')


class ParamsInfo(object):
    def __init__(self, params):
        n_params = len(params)
        data_ptr = np.empty(n_params, dtype=np.int64)
        grad_ptr = np.empty(n_params, dtype=np.int64)
        v_ptr = np.empty(n_params, dtype=np.int64)
        fp32_data_ptr = np.empty(n_params, dtype=np.int64)
        data_dtype = np.empty(n_params, dtype=np.int32)
        size_csum = np.empty(n_params+1, dtype=np.int32)
        size_csum[0] = 0
        for i, param in enumerate(params):
            data = param.data
            grad = param.grad
            v = param.update_rule.state['v']
            data_ptr[i] = data.data.ptr
            grad_ptr[i] = grad.data.ptr
            v_ptr[i] = v.data.ptr
            assert(data.dtype == grad.dtype)
            data_dtype[i] = _get_dtype_id(data.dtype)
            if param.update_rule._fp32_param is None:
                fp32_data_ptr[i] = 0
            else:
                fp32_data = param.update_rule._fp32_param.data
                assert(fp32_data.dtype == np.float32)
                assert(v.dtype == np.float32)
                assert(data.dtype == np.float16)
                fp32_data_ptr[i] = fp32_data.data.ptr
            size_csum[i+1] = size_csum[i] + data.size
        self.n_elems = size_csum[n_params]
        self.n_params = n_params
        self.data_ptr = cuda.cupy.asarray(data_ptr)
        self.grad_ptr = cuda.cupy.asarray(grad_ptr)
        self.v_ptr = cuda.cupy.asarray(v_ptr)
        self.fp32_data_ptr = cuda.cupy.asarray(fp32_data_ptr)
        self.data_dtype = cuda.cupy.asarray(data_dtype)
        self.size_csum = cuda.cupy.asarray(size_csum)


def _get_dtype_id(dtype):
    if dtype == np.float16:
        return 6  # nccl.NCCL_FLOAT16
    elif dtype == np.float32:
        return 7  # nccl.NCCL_FLOAT32
    else:
        raise ValueError(
            'dtype must be float16 or float32.')


class MomentumSGD(optimizer.GradientMethod):

    """Momentum SGD optimizer.

    Args:
        lr (float): Learning rate.
        momentum (float): Exponential decay rate of the first order moment.

    """

    def __init__(self, lr=_default_hyperparam.lr,
                 momentum=_default_hyperparam.momentum):
        super(MomentumSGD, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.momentum = momentum

    lr = optimizer.HyperparameterProxy('lr')
    momentum = optimizer.HyperparameterProxy('momentum')

    def create_update_rule(self):
        return MomentumSGDRule(self.hyperparam)
