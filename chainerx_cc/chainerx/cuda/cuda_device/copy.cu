#include "chainerx/cuda/cuda_device.h"

#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/creation.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename InT, typename OutT>
struct CopyImpl {
    using InCudaType = cuda_internal::DataType<InT>;
    using OutCudaType = cuda_internal::DataType<OutT>;
    __device__ void operator()(int64_t /*i*/, InCudaType a, OutCudaType& out) { out = static_cast<OutCudaType>(a); }
};

class CudaCopyKernel : public CopyKernel {
public:
    void Call(const Array& a, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};
        auto do_astype = [&](auto in_pt, auto out_pt) {
            using InT = typename decltype(in_pt)::type;
            using OutT = typename decltype(out_pt)::type;
            Elementwise<const InT, OutT>(CopyImpl<InT, OutT>{}, a, out);
        };
        VisitDtype(out.dtype(), [&](auto out_pt) { VisitDtype(a.dtype(), do_astype, out_pt); });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(CopyKernel, CudaCopyKernel);

}  // namespace
}  // namespace cuda
}  // namespace chainerx
