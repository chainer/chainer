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
cusolverStatus_t Getrf(cusolverDnHandle_t /*handle*/, int /*m*/, int /*n*/, T* /*a*/, int /*lda*/, T* /*workspace*/, int* /*devipiv*/, int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by getrf (LU)"};
}

template <typename T>
cusolverStatus_t Getrs(cusolverDnHandle_t /*handle*/, cublasOperation_t /*trans*/, int /*n*/, int /*nrhs*/, T* /*a*/, int /*lda*/, int* /*devipiv*/, T* /*b*/, int /*ldb*/, int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by getrs (Solve)"};
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
cusolverStatus_t Getrs<double>(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, double* a, int lda, int* devipiv, double* b, int ldb, int* devinfo) {
    return cusolverDnDgetrs(handle, trans, n, nrhs, a, lda, devipiv, b, ldb, devinfo);
}

template <>
cusolverStatus_t Getrs<float>(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, float* a, int lda, int* devipiv, float* b, int ldb, int* devinfo) {
    return cusolverDnSgetrs(handle, trans, n, nrhs, a, lda, devipiv, b, ldb, devinfo);
}

template <typename T>
void SolveImpl(const Array& a, const Array& b, const Array& out) {
    Device& device = a.device();
    Dtype dtype = a.dtype();

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

    Array lu_matrix = Empty(a.shape(), dtype, device);
    device.backend().CallKernel<CopyKernel>(a.Transpose(), lu_matrix);
    T* lu_ptr = static_cast<T*>(internal::GetRawOffsetData(lu_matrix));

    int m = a.shape()[0];

    Array ipiv = Empty(Shape{m}, Dtype::kInt32, device);
    int* ipiv_ptr = static_cast<int*>(internal::GetRawOffsetData(ipiv));

    int buffersize = 0;
    device_internals.cusolverdn_handle().Call(GetrfBuffersize<T>, m, m, lu_ptr, m, &buffersize);

    Array work = Empty(Shape{buffersize}, dtype, device);
    T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

    std::shared_ptr<void> devinfo = device.Allocate(sizeof(int));

    device_internals.cusolverdn_handle().Call(Getrf<T>, m, m, lu_ptr, m, work_ptr, ipiv_ptr, static_cast<int*>(devinfo.get()));

    int devinfo_h = 0;
    Device& native_device = dynamic_cast<native::NativeDevice&>(GetDefaultContext().GetDevice({"native", 0}));
    device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
    if (devinfo_h != 0) {
        throw ChainerxError{"Unsuccessfull getrf (LU) execution. Info = ", devinfo_h};
    }

    device.backend().CallKernel<CopyKernel>(b, out);
    T* out_ptr = static_cast<T*>(internal::GetRawOffsetData(out));

    device_internals.cusolverdn_handle().Call(
            Getrs<T>, CUBLAS_OP_N, m, m, lu_ptr, m, ipiv_ptr, out_ptr, m, static_cast<int*>(devinfo.get()));

    device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
    if (devinfo_h != 0) {
        throw ChainerxError{"Unsuccessfull getrs (Solve) execution. Info = ", devinfo_h};
    }
}

}  // namespace

class CudaSolveKernel : public SolveKernel {
public:
    void Call(const Array& a, const Array& b, const Array& out) override {
        Device& device = a.device();
        // Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

        VisitFloatingPointDtype(a.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            SolveImpl<T>(a, b, out);
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
        device.backend().CallKernel<SolveKernel>(a.Transpose(), b, out);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(InverseKernel, CudaInverseKernel);

}  // namespace cuda
}  // namespace chainerx
