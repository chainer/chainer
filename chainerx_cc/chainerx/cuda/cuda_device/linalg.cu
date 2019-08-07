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
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"
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
cusolverStatus_t GesvdBuffersize(cusolverDnHandle_t /*handle*/, int /*m*/, int /*n*/, int* /*lwork*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by gesvd (SVD)"};
}

template <typename T>
cusolverStatus_t Gesvd(
        cusolverDnHandle_t /*handle*/,
        signed char /*jobu*/,
        signed char /*jobvt*/,
        int /*m*/,
        int /*n*/,
        T* /*a*/,
        int /*lda*/,
        T* /*s*/,
        T* /*u*/,
        int /*ldu*/,
        T* /*vt*/,
        int /*ldvt*/,
        T* /*work*/,
        int /*lwork*/,
        T* /*rwork*/,
        int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by gesvd (SVD)"};
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
cusolverStatus_t GesvdBuffersize<double>(cusolverDnHandle_t handle, int m, int n, int* lwork) {
    return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
}

template <>
cusolverStatus_t GesvdBuffersize<float>(cusolverDnHandle_t handle, int m, int n, int* lwork) {
    return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
}

template <>
cusolverStatus_t Gesvd<double>(
        cusolverDnHandle_t handle,
        signed char jobu,
        signed char jobvt,
        int m,
        int n,
        double* a,
        int lda,
        double* s,
        double* u,
        int ldu,
        double* vt,
        int ldvt,
        double* work,
        int lwork,
        double* rwork,
        int* devinfo) {
    return cusolverDnDgesvd(handle, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, devinfo);
}

template <>
cusolverStatus_t Gesvd<float>(
        cusolverDnHandle_t handle,
        signed char jobu,
        signed char jobvt,
        int m,
        int n,
        float* a,
        int lda,
        float* s,
        float* u,
        int ldu,
        float* vt,
        int ldvt,
        float* work,
        int lwork,
        float* rwork,
        int* devinfo) {
    return cusolverDnSgesvd(handle, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, devinfo);
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

class CudaSvdKernel : public SvdKernel {
public:
    void Call(const Array& a, const Array& u, const Array& s, const Array& vt, bool full_matrices) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);

        bool compute_uv = u.shape()[0] != 0 && vt.shape()[0] != 0;

        // cuSOLVER assumes arrays are in column-major order.
        // In order to avoid transposing the input matrix, matrix dimensions are swapped.
        // Since the input is assumed to be transposed, it is necessary to
        // swap the pointers to u and vt matrices when calling Gesvd.
        int64_t n = a.shape()[0];
        int64_t m = a.shape()[1];
        int64_t k = std::min(m, n);

        Array x = EmptyLike(a, device);
        Array u_temp{};
        Array vt_temp{};
        bool trans_flag;

        // Remark: gesvd only supports m>=n.
        // See: https://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-gesvd
        // Therefore for the case m<n we calculuate svd of transposed matrix,
        // instead of calculating svd(A) = U S V^T, we compute svd(A^T) = V S U^T
        if (m >= n) {
            device.backend().CallKernel<CopyKernel>(a, x);
            trans_flag = false;
        } else {
            m = a.shape()[0];
            n = a.shape()[1];
            x = x.Reshape(Shape{n, m});
            device.backend().CallKernel<CopyKernel>(a.Transpose(), x);
            trans_flag = true;

            // Temporary arrays for u, vt are needed to store transposed results
            Shape u_shape;
            Shape vt_shape;
            if (compute_uv) {
                if (full_matrices) {
                    u_shape = Shape{m, m};
                    vt_shape = Shape{n, n};
                } else {
                    u_shape = Shape{k, m};
                    vt_shape = Shape{n, k};
                }
            } else {
                u_shape = Shape{0};
                vt_shape = Shape{0};
            }
            u_temp = Empty(u_shape, dtype, device);
            vt_temp = Empty(vt_shape, dtype, device);
        }

        int64_t ldu = m;
        int64_t ldvt = full_matrices ? n : k;

        auto svd_impl = [&](auto pt) {
            using T = typename decltype(pt)::type;
            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            auto x_ptr = static_cast<T*>(internal::GetRawOffsetData(x));
            auto s_ptr = static_cast<T*>(internal::GetRawOffsetData(s));
            auto u_ptr = static_cast<T*>(internal::GetRawOffsetData(u));
            auto vt_ptr = static_cast<T*>(internal::GetRawOffsetData(vt));
            if (trans_flag) {
                u_ptr = static_cast<T*>(internal::GetRawOffsetData(vt_temp));
                vt_ptr = static_cast<T*>(internal::GetRawOffsetData(u_temp));
            }

            std::shared_ptr<void> devInfo = device.Allocate(sizeof(int));

            int buffersize = 0;
            device_internals.cusolverdn_handle().Call(GesvdBuffersize<T>, m, n, &buffersize);

            Array work = Empty(Shape{buffersize}, dtype, device);
            auto work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            signed char job;
            if (compute_uv) {
                job = full_matrices ? 'A' : 'S';
            } else {
                job = 'N';
            }

            // When calling Gesvd pointers to u and vt are swapped instead of transposing the input matrix.
            device_internals.cusolverdn_handle().Call(
                    Gesvd<T>,
                    job,
                    job,
                    m,
                    n,
                    x_ptr,
                    m,
                    s_ptr,
                    vt_ptr,
                    ldu,
                    u_ptr,
                    ldvt,
                    work_ptr,
                    buffersize,
                    nullptr,
                    static_cast<int*>(devInfo.get()));

            int devInfo_h = 0;
            Device& native_device = GetDefaultContext().GetDevice({"native", 0});
            device.MemoryCopyTo(&devInfo_h, devInfo.get(), sizeof(int), native_device);
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessful gesvd (SVD) execution. Info = ", devInfo_h};
            }

            if (trans_flag) {
                device.backend().CallKernel<CopyKernel>(u_temp.Transpose(), u);
                device.backend().CallKernel<CopyKernel>(vt_temp.Transpose(), vt);
            }
        };

        VisitFloatingPointDtype(dtype, svd_impl);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(SvdKernel, CudaSvdKernel);

}  // namespace cuda
}  // namespace chainerx
