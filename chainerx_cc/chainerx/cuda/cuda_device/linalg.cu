#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cusolver.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_device.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/linalg.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
cusolverStatus_t GeqrfBufferSize(cusolverDnHandle_t /*handle*/, int /*m*/, int /*n*/, T* /*a*/, int /*lda*/, int* /*lwork*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by geqrf (QR)"};
}

template <typename T>
cusolverStatus_t Geqrf(
        cusolverDnHandle_t /*handle*/,
        int /*m*/,
        int /*n*/,
        T* /*a*/,
        int /*lda*/,
        T* /*tau*/,
        T* /*workspace*/,
        int /*lwork*/,
        int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by geqrf (QR)"};
}

template <typename T>
cusolverStatus_t OrgqrBufferSize(
        cusolverDnHandle_t /*handle*/, int /*m*/, int /*n*/, int /*k*/, T* /*a*/, int /*lda*/, T* /*tau*/, int* /*lwork*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by orgqr (QR)"};
}

template <typename T>
cusolverStatus_t Orgqr(
        cusolverDnHandle_t /*handle*/,
        int /*m*/,
        int /*n*/,
        int /*k*/,
        T* /*a*/,
        int /*lda*/,
        T* /*tau*/,
        T* /*work*/,
        int /*lwork*/,
        int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by orgqr (QR)"};
}

template <>
cusolverStatus_t GeqrfBufferSize<double>(cusolverDnHandle_t handle, int m, int n, double* a, int lda, int* lwork) {
    return cusolverDnDgeqrf_bufferSize(handle, m, n, a, lda, lwork);
}

template <>
cusolverStatus_t GeqrfBufferSize<float>(cusolverDnHandle_t handle, int m, int n, float* a, int lda, int* lwork) {
    return cusolverDnSgeqrf_bufferSize(handle, m, n, a, lda, lwork);
}

template <>
cusolverStatus_t Geqrf<double>(
        cusolverDnHandle_t handle, int m, int n, double* a, int lda, double* tau, double* workspace, int lwork, int* devinfo) {
    return cusolverDnDgeqrf(handle, m, n, a, lda, tau, workspace, lwork, devinfo);
}

template <>
cusolverStatus_t Geqrf<float>(
        cusolverDnHandle_t handle, int m, int n, float* a, int lda, float* tau, float* workspace, int lwork, int* devinfo) {
    return cusolverDnSgeqrf(handle, m, n, a, lda, tau, workspace, lwork, devinfo);
}

template <>
cusolverStatus_t OrgqrBufferSize<double>(cusolverDnHandle_t handle, int m, int n, int k, double* a, int lda, double* tau, int* lwork) {
    return cusolverDnDorgqr_bufferSize(handle, m, n, k, a, lda, tau, lwork);
}

template <>
cusolverStatus_t OrgqrBufferSize<float>(cusolverDnHandle_t handle, int m, int n, int k, float* a, int lda, float* tau, int* lwork) {
    return cusolverDnSorgqr_bufferSize(handle, m, n, k, a, lda, tau, lwork);
}

template <>
cusolverStatus_t Orgqr<double>(
        cusolverDnHandle_t handle, int m, int n, int k, double* a, int lda, double* tau, double* work, int lwork, int* devinfo) {
    return cusolverDnDorgqr(handle, m, n, k, a, lda, tau, work, lwork, devinfo);
}

template <>
cusolverStatus_t Orgqr<float>(
        cusolverDnHandle_t handle, int m, int n, int k, float* a, int lda, float* tau, float* work, int lwork, int* devinfo) {
    return cusolverDnSorgqr(handle, m, n, k, a, lda, tau, work, lwork, devinfo);
}

template <typename T>
std::tuple<Array, Array> QRImpl(const Array& a, QRMode mode) {
    Device& device = a.device();
    Dtype dtype = a.dtype();

    int64_t m = a.shape()[0];
    int64_t n = a.shape()[1];
    int64_t mn = std::min(m, n);

    Array Q = Empty(Shape{0}, dtype, device);
    Array R = a.Transpose().Copy();  // QR decomposition is done in-place
    Array tau = Empty(Shape{mn}, dtype, device);

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

    T* r_ptr = static_cast<T*>(internal::GetRawOffsetData(R));
    T* tau_ptr = static_cast<T*>(internal::GetRawOffsetData(tau));

    std::shared_ptr<void> devInfo = device.Allocate(sizeof(int));

    int buffersize_geqrf = 0;
    device_internals.cusolverdn_handle().Call(GeqrfBufferSize<T>, m, n, r_ptr, n, &buffersize_geqrf);

    Array work = Empty(Shape{buffersize_geqrf}, dtype, device);
    T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

    device_internals.cusolverdn_handle().Call(
            Geqrf<T>, m, n, r_ptr, m, tau_ptr, work_ptr, buffersize_geqrf, static_cast<int*>(devInfo.get()));

    int devInfo_h = 0;
    Device& native_device = GetDefaultContext().GetDevice({"native", 0});
    device.MemoryCopyTo(&devInfo_h, devInfo.get(), sizeof(int), native_device);
    if (devInfo_h != 0) {
        throw ChainerxError{"Unsuccessful geqrf (QR) execution. Info = ", devInfo_h};
    }

    if (mode == QRMode::r) {
        R = R.At(std::vector<ArrayIndex>{Slice{}, Slice{0, mn}}).Transpose();  // R = R[:, 0:mn].T
        R = Triu(R, 0);
        return std::make_tuple(std::move(Q), std::move(R));
    }

    if (mode == QRMode::raw) {
        return std::make_tuple(std::move(R), std::move(tau));
    }

    int64_t mc;
    if (mode == QRMode::complete && m > n) {
        mc = m;
        Q = Empty(Shape{m, m}, dtype, device);
    } else {
        mc = mn;
        Q = Empty(Shape{n, m}, dtype, device);
    }

    device.backend().CallKernel<CopyKernel>(R, Q.At(std::vector<ArrayIndex>{Slice{0, n}, Slice{}}));  // Q[0:n, :] = R
    T* q_ptr = static_cast<T*>(internal::GetRawOffsetData(Q));

    int buffersize_orgqr = 0;
    device_internals.cusolverdn_handle().Call(OrgqrBufferSize<T>, m, mc, mn, q_ptr, m, tau_ptr, &buffersize_orgqr);

    work = Empty(Shape{buffersize_orgqr}, dtype, device);

    device_internals.cusolverdn_handle().Call(
            Orgqr<T>, m, mc, mn, q_ptr, m, tau_ptr, work_ptr, buffersize_orgqr, static_cast<int*>(devInfo.get()));

    device.MemoryCopyTo(&devInfo_h, devInfo.get(), sizeof(int), native_device);
    if (devInfo_h != 0) {
        throw ChainerxError{"Unsuccessful orgqr (QR) execution. Info = ", devInfo_h};
    }

    // .Copy() is needed to have correct strides
    Q = Q.At(std::vector<ArrayIndex>{Slice{0, mc}, Slice{}}).Transpose().Copy();  // Q = Q[0:mc, :].T
    R = R.At(std::vector<ArrayIndex>{Slice{}, Slice{0, mc}}).Transpose();  // R = R[:, 0:mc].T
    R = Triu(R, 0);
    return std::make_tuple(std::move(Q), std::move(R));
}

}  // namespace

class CudaQRKernel : public QRKernel {
public:
    std::tuple<Array, Array> Call(const Array& a, QRMode mode = QRMode::reduced) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);

        return VisitFloatingPointDtype(dtype, [&](auto pt) -> std::tuple<Array, Array> {
            using T = typename decltype(pt)::type;
            return QRImpl<T>(a, mode);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(QRKernel, CudaQRKernel);

}  // namespace cuda
}  // namespace chainerx
