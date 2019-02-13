#include "chainerx/cuda/cuda_device.h"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "chainerx/array.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/elementwise.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace cuda {

namespace {

template <typename T>
struct AdamImpl {
    __device__ void operator()(int64_t /*i*/, T grad, T& param, T& m, T& v) {
        m += one_minus_beta1 * (grad - m);
        v += one_minus_beta2 * (grad * grad - v);
        vhat = std::max(vhat, v);
        param -= eta * (alpha_t * m / (std::sqrt(vhat) + eps) + weight_decay_rate * param);
    }
    T alpha_t;
    T one_minus_beta1;
    T one_minus_beta2;
    T eps;
    T eta;
    T weight_decay_rate;
};

}  // namespace

void CudaDevice::Adam(
        const Array& grad,
        Scalar alpha,
        Scalar beta1,
        Scalar beta2,
        Scalar eps,
        Scalar eta,
        Scalar weight_decay_rate,
        const Array& param,
        const Array& m,
        const Array& t) {
    CheckDevicesCompatible(grad, param, m, t);
    CudaSetDeviceScope scope{index()};
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(
                AdamImpl<T>{static_cast<T>(alpha),
                            static_cast<T>(1 - beta1),
                            static_cast<T>(1 - beta2),
                            static_cast<T>(eps),
                            static_cast<T>(eta),
                            static_cast<T>(weight_decay_rate)},
                grad,
                param,
                m,
                t);
    });
}

}  // namespace cuda
}  // namespace chainerx
