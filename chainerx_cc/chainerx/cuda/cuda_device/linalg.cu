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
cusolverStatus_t GetrfBuffersize(cusolverDnHandle_t /*handle*/, int /*m*/, int /*n*/, T* /*a*/, int /*lda*/, int* /*lwork*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by getrf (LU)"};
}

template <typename T>
cusolverStatus_t Getrf(
        cusolverDnHandle_t /*handle*/, int /*m*/, int /*n*/, T* /*a*/, int /*lda*/, T* /*workspace*/, int* /*devipiv*/, int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by getrf (LU)"};
}

template <typename T>
cusolverStatus_t Getrs(
        cusolverDnHandle_t /*handle*/,
        cublasOperation_t /*trans*/,
        int /*n*/,
        int /*nrhs*/,
        T* /*a*/,
        int /*lda*/,
        int* /*devipiv*/,
        T* /*b*/,
        int /*ldb*/,
        int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by getrs (Solve)"};
}

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
cusolverStatus_t GetrfBuffersize<double>(cusolverDnHandle_t handle, int m, int n, double* a, int lda, int* lwork) {
    return cusolverDnDgetrf_bufferSize(handle, m, n, a, lda, lwork);
}

template <>
cusolverStatus_t GetrfBuffersize<float>(cusolverDnHandle_t handle, int m, int n, float* a, int lda, int* lwork) {
    return cusolverDnSgetrf_bufferSize(handle, m, n, a, lda, lwork);
}

template <>
cusolverStatus_t Getrf<double>(cusolverDnHandle_t handle, int m, int n, double* a, int lda, double* workspace, int* devipiv, int* devinfo) {
    return cusolverDnDgetrf(handle, m, n, a, lda, workspace, devipiv, devinfo);
}

template <>
cusolverStatus_t Getrf<float>(cusolverDnHandle_t handle, int m, int n, float* a, int lda, float* workspace, int* devipiv, int* devinfo) {
    return cusolverDnSgetrf(handle, m, n, a, lda, workspace, devipiv, devinfo);
}

template <>
cusolverStatus_t Getrs<double>(
        cusolverDnHandle_t handle,
        cublasOperation_t trans,
        int n,
        int nrhs,
        double* a,
        int lda,
        int* devipiv,
        double* b,
        int ldb,
        int* devinfo) {
    return cusolverDnDgetrs(handle, trans, n, nrhs, a, lda, devipiv, b, ldb, devinfo);
}

template <>
cusolverStatus_t Getrs<float>(
        cusolverDnHandle_t handle,
        cublasOperation_t trans,
        int n,
        int nrhs,
        float* a,
        int lda,
        int* devipiv,
        float* b,
        int ldb,
        int* devinfo) {
    return cusolverDnSgetrs(handle, trans, n, nrhs, a, lda, devipiv, b, ldb, devinfo);
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
void SolveImpl(const Array& a, const Array& b, const Array& out) {
    Device& device = a.device();
    Dtype dtype = a.dtype();

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

    Array lu_matrix = Empty(a.shape(), dtype, device);
    device.backend().CallKernel<CopyKernel>(a.Transpose(), lu_matrix);
    auto lu_ptr = static_cast<T*>(internal::GetRawOffsetData(lu_matrix));

    int64_t m = a.shape()[0];
    int64_t nrhs = 1;
    if (b.ndim() == 2) {
        nrhs = b.shape()[1];
    }

    Array ipiv = Empty(Shape{m}, Dtype::kInt32, device);
    auto ipiv_ptr = static_cast<int*>(internal::GetRawOffsetData(ipiv));

    int buffersize = 0;
    device_internals.cusolverdn_handle().Call(GetrfBuffersize<T>, m, m, lu_ptr, m, &buffersize);

    Array work = Empty(Shape{buffersize}, dtype, device);
    auto work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

    std::shared_ptr<void> devinfo = device.Allocate(sizeof(int));

    device_internals.cusolverdn_handle().Call(Getrf<T>, m, m, lu_ptr, m, work_ptr, ipiv_ptr, static_cast<int*>(devinfo.get()));

    int devinfo_h = 0;
    Device& native_device = GetDefaultContext().GetDevice({"native", 0});
    device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
    if (devinfo_h != 0) {
        throw ChainerxError{"Unsuccessful getrf (LU) execution. Info = ", devinfo_h};
    }

    Array out_transposed = b.Transpose().Copy();
    auto out_ptr = static_cast<T*>(internal::GetRawOffsetData(out_transposed));

    device_internals.cusolverdn_handle().Call(
            Getrs<T>, CUBLAS_OP_N, m, nrhs, lu_ptr, m, ipiv_ptr, out_ptr, m, static_cast<int*>(devinfo.get()));

    device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
    if (devinfo_h != 0) {
        throw ChainerxError{"Unsuccessful getrs (Solve) execution. Info = ", devinfo_h};
    }

    device.backend().CallKernel<CopyKernel>(out_transposed.Transpose(), out);
}

template <typename T>
void QrImpl(const Array& a, const Array& q, const Array& r, const Array& tau, QrMode mode) {
    Device& device = a.device();
    Dtype dtype = a.dtype();

    int64_t m = a.shape()[0];
    int64_t n = a.shape()[1];
    int64_t k = std::min(m, n);

    Array R = a.Transpose().Copy();  // QR decomposition is done in-place

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

    auto r_ptr = static_cast<T*>(internal::GetRawOffsetData(R));
    auto tau_ptr = static_cast<T*>(internal::GetRawOffsetData(tau));

    std::shared_ptr<void> devInfo = device.Allocate(sizeof(int));

    int buffersize_geqrf = 0;
    device_internals.cusolverdn_handle().Call(GeqrfBufferSize<T>, m, n, r_ptr, n, &buffersize_geqrf);

    Array work = Empty(Shape{buffersize_geqrf}, dtype, device);
    auto work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

    device_internals.cusolverdn_handle().Call(
            Geqrf<T>, m, n, r_ptr, m, tau_ptr, work_ptr, buffersize_geqrf, static_cast<int*>(devInfo.get()));

    int devInfo_h = 0;
    Device& native_device = GetDefaultContext().GetDevice({"native", 0});
    device.MemoryCopyTo(&devInfo_h, devInfo.get(), sizeof(int), native_device);
    if (devInfo_h != 0) {
        throw ChainerxError{"Unsuccessful geqrf (QR) execution. Info = ", devInfo_h};
    }

    if (mode == QrMode::kR) {
        R = R.At(std::vector<ArrayIndex>{Slice{}, Slice{0, k}}).Transpose();  // R = R[:, 0:k].T
        R = Triu(R, 0);
        device.backend().CallKernel<CopyKernel>(R, r);
        return;
    }

    if (mode == QrMode::kRaw) {
        device.backend().CallKernel<CopyKernel>(R, r);
        return;
    }

    int64_t mc;
    Shape q_shape{0};
    if (mode == QrMode::kComplete && m > n) {
        mc = m;
        q_shape = Shape{m, m};
    } else {
        mc = k;
        q_shape = Shape{n, m};
    }
    Array Q = Empty(q_shape, dtype, device);

    device.backend().CallKernel<CopyKernel>(R, Q.At(std::vector<ArrayIndex>{Slice{0, n}, Slice{}}));  // Q[0:n, :] = R
    auto q_ptr = static_cast<T*>(internal::GetRawOffsetData(Q));

    int buffersize_orgqr = 0;
    device_internals.cusolverdn_handle().Call(OrgqrBufferSize<T>, m, mc, k, q_ptr, m, tau_ptr, &buffersize_orgqr);

    work = Empty(Shape{buffersize_orgqr}, dtype, device);

    device_internals.cusolverdn_handle().Call(
            Orgqr<T>, m, mc, k, q_ptr, m, tau_ptr, work_ptr, buffersize_orgqr, static_cast<int*>(devInfo.get()));

    device.MemoryCopyTo(&devInfo_h, devInfo.get(), sizeof(int), native_device);
    if (devInfo_h != 0) {
        throw ChainerxError{"Unsuccessful orgqr (QR) execution. Info = ", devInfo_h};
    }

    Q = Q.At(std::vector<ArrayIndex>{Slice{0, mc}, Slice{}}).Transpose();  // Q = Q[0:mc, :].T
    R = R.At(std::vector<ArrayIndex>{Slice{}, Slice{0, mc}}).Transpose();  // R = R[:, 0:mc].T
    R = Triu(R, 0);

    device.backend().CallKernel<CopyKernel>(Q, q);
    device.backend().CallKernel<CopyKernel>(R, r);
}

}  // namespace

class CudaSolveKernel : public SolveKernel {
public:
    void Call(const Array& a, const Array& b, const Array& out) override {
        Device& device = a.device();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            SolveImpl<T>(a.dtype() == out.dtype() ? a : a.AsType(out.dtype()), b.dtype() == out.dtype() ? b : b.AsType(out.dtype()), out);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(SolveKernel, CudaSolveKernel);

class CudaInverseKernel : public InverseKernel {
public:
    void Call(const Array& a, const Array& out) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

        // There is LAPACK routine ``getri`` for computing the inverse of an LU-factored matrix,
        // but cuSOLVER does not have it implemented, therefore inverse is obtained with ``getrs``
        // inv(A) == solve(A, Identity)
        Array b = Identity(a.shape()[0], dtype, device);
        device.backend().CallKernel<SolveKernel>(a, b, out);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(InverseKernel, CudaInverseKernel);

class CudaQrKernel : public QrKernel {
public:
    void Call(const Array& a, const Array& q, const Array& r, const Array& tau, QrMode mode) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);

        VisitFloatingPointDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            QrImpl<T>(a, q, r, tau, mode);
        });
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(QrKernel, CudaQrKernel);

}  // namespace cuda
}  // namespace chainerx
