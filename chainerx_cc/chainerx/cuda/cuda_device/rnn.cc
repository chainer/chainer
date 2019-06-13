#include "chainerx/cuda/cuda_device.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/cuda/cuda_rnn.h"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/kernels/rnn.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {
namespace {

class CudaRnnKernel : public RnnKernel {
public:
    std::vector<std::vector<Array>> Call(
        int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        const int8_t bidirectional,
        const int8_t mode) override {

        CudaDevice& device = dynamic_cast<CudaDevice&>(hx.device());
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        return device_internals.cuda_rnn().n_step_rnn(device, n_layers, hx, cx, ws, bs, xs, bidirectional, mode);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(RnnKernel, CudaRnnKernel);

class CudaRnnBackwardKernel : public RnnBackwardKernel {
public:
    std::vector<std::vector<Array>> Call(
        int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        Array dhy,
        Array dcy,
        std::vector<Array> ys,
        std::vector<Array> dys,
        const int8_t bidirectional,
        const int8_t mode
        ) override {
        
        CudaDevice& device = dynamic_cast<CudaDevice&>(hx.device());
        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(device);
        return device_internals.cuda_rnn().n_step_rnn_backward(device, n_layers, hx, cx, ws, bs, xs, dhy, dcy, ys, dys, bidirectional, mode);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(RnnBackwardKernel, CudaRnnBackwardKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
