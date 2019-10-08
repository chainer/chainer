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

template <typename T>
cusolverStatus_t PotrfBuffersize(
        cusolverDnHandle_t /*handle*/, cublasFillMode_t /*uplo*/, int /*n*/, T* /*a*/, int /*lda*/, int* /*lwork*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by potrf (Cholesky)"};
}

template <typename T>
cusolverStatus_t Potrf(
        cusolverDnHandle_t /*handle*/,
        cublasFillMode_t /*uplo*/,
        int /*n*/,
        T* /*a*/,
        int /*lda*/,
        T* /*workspace*/,
        int /*lwork*/,
        int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by potrf (Cholesky)"};
}

template <typename T>
cusolverStatus_t SyevdBuffersize(
        cusolverDnHandle_t /*handle*/,
        cusolverEigMode_t /*jobz*/,
        cublasFillMode_t /*uplo*/,
        int /*n*/,
        T* /*a*/,
        int /*lda*/,
        T* /*w*/,
        int* /*lwork*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by syevd (Eigen)"};
}

template <typename T>
cusolverStatus_t Syevd(
        cusolverDnHandle_t /*handle*/,
        cusolverEigMode_t /*jobz*/,
        cublasFillMode_t /*uplo*/,
        int /*n*/,
        T* /*a*/,
        int /*lda*/,
        T* /*w*/,
        T* /*work*/,
        int /*lwork*/,
        int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by syevd (Eigen)"};
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

template <>
cusolverStatus_t PotrfBuffersize<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* a, int lda, int* lwork) {
    return cusolverDnDpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
}

template <>
cusolverStatus_t PotrfBuffersize<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* a, int lda, int* lwork) {
    return cusolverDnSpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
}

template <>
cusolverStatus_t Potrf<double>(
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* a, int lda, double* workspace, int lwork, int* devinfo) {
    return cusolverDnDpotrf(handle, uplo, n, a, lda, workspace, lwork, devinfo);
}

template <>
cusolverStatus_t Potrf<float>(
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* a, int lda, float* workspace, int lwork, int* devinfo) {
    return cusolverDnSpotrf(handle, uplo, n, a, lda, workspace, lwork, devinfo);
}

template <>
cusolverStatus_t SyevdBuffersize<double>(
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double* a, int lda, double* w, int* lwork) {
    return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, a, lda, w, lwork);
}

template <>
cusolverStatus_t SyevdBuffersize<float>(
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float* a, int lda, float* w, int* lwork) {
    return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, a, lda, w, lwork);
}

template <>
cusolverStatus_t Syevd<double>(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        cublasFillMode_t uplo,
        int n,
        double* a,
        int lda,
        double* w,
        double* work,
        int lwork,
        int* devinfo) {
    return cusolverDnDsyevd(handle, jobz, uplo, n, a, lda, w, work, lwork, devinfo);
}

template <>
cusolverStatus_t Syevd<float>(
        cusolverDnHandle_t handle,
        cusolverEigMode_t jobz,
        cublasFillMode_t uplo,
        int n,
        float* a,
        int lda,
        float* w,
        float* work,
        int lwork,
        int* devinfo) {
    return cusolverDnSsyevd(handle, jobz, uplo, n, a, lda, w, work, lwork, devinfo);
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
    int64_t lda = std::max(int64_t{1}, m);
    int64_t nrhs = 1;
    if (b.ndim() == 2) {
        nrhs = b.shape()[1];
    }

    Array ipiv = Empty(Shape{m}, Dtype::kInt32, device);
    auto ipiv_ptr = static_cast<int*>(internal::GetRawOffsetData(ipiv));

    int buffersize = 0;
    device_internals.cusolverdn_handle().Call(GetrfBuffersize<T>, m, m, lu_ptr, lda, &buffersize);

    Array work = Empty(Shape{buffersize}, dtype, device);
    auto work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

    std::shared_ptr<void> devinfo = device.Allocate(sizeof(int));

    device_internals.cusolverdn_handle().Call(Getrf<T>, m, m, lu_ptr, lda, work_ptr, ipiv_ptr, static_cast<int*>(devinfo.get()));

    int devinfo_h = 0;
    Device& native_device = GetDefaultContext().GetDevice({"native", 0});
    device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
    if (devinfo_h != 0) {
        throw ChainerxError{"Unsuccessful getrf (LU) execution. Info = ", devinfo_h};
    }

    Array out_transposed = b.Transpose().Copy();
    auto out_ptr = static_cast<T*>(internal::GetRawOffsetData(out_transposed));

    device_internals.cusolverdn_handle().Call(
            Getrs<T>, CUBLAS_OP_N, m, nrhs, lu_ptr, lda, ipiv_ptr, out_ptr, lda, static_cast<int*>(devinfo.get()));

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
    int64_t lda = std::max(int64_t{1}, m);

    // cuSOLVER does not return correct result in this case and older versions of cuSOLVER (<10.1)
    // might not work well with zero-sized arrays therefore it's better to return earlier
    if (a.shape().GetTotalSize() == 0) {
        if (mode == QrMode::kComplete) {
            device.backend().CallKernel<IdentityKernel>(q);
        }
        return;
    }

    Array r_temp = a.Transpose().Copy();  // QR decomposition is done in-place

    cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

    auto r_ptr = static_cast<T*>(internal::GetRawOffsetData(r_temp));
    auto tau_ptr = static_cast<T*>(internal::GetRawOffsetData(tau));

    std::shared_ptr<void> devinfo = device.Allocate(sizeof(int));

    int buffersize_geqrf = 0;
    device_internals.cusolverdn_handle().Call(GeqrfBufferSize<T>, m, n, r_ptr, lda, &buffersize_geqrf);

    Array work = Empty(Shape{buffersize_geqrf}, dtype, device);
    auto work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

    device_internals.cusolverdn_handle().Call(
            Geqrf<T>, m, n, r_ptr, lda, tau_ptr, work_ptr, buffersize_geqrf, static_cast<int*>(devinfo.get()));

    int devinfo_h = 0;
    Device& native_device = GetDefaultContext().GetDevice({"native", 0});
    device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
    if (devinfo_h != 0) {
        throw ChainerxError{"Unsuccessful geqrf (QR) execution. Info = ", devinfo_h};
    }

    if (mode == QrMode::kR) {
        r_temp = r_temp.At(std::vector<ArrayIndex>{Slice{}, Slice{0, k}}).Transpose();  // R = R[:, 0:k].T
        r_temp = Triu(r_temp, 0);
        device.backend().CallKernel<CopyKernel>(r_temp, r);
        return;
    }

    if (mode == QrMode::kRaw) {
        device.backend().CallKernel<CopyKernel>(r_temp, r);
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
    Array q_temp = Empty(q_shape, dtype, device);

    device.backend().CallKernel<CopyKernel>(r_temp, q_temp.At(std::vector<ArrayIndex>{Slice{0, n}, Slice{}}));  // Q[0:n, :] = R
    auto q_ptr = static_cast<T*>(internal::GetRawOffsetData(q_temp));

    int buffersize_orgqr = 0;
    device_internals.cusolverdn_handle().Call(OrgqrBufferSize<T>, m, mc, k, q_ptr, lda, tau_ptr, &buffersize_orgqr);

    Array work_orgqr = Empty(Shape{buffersize_orgqr}, dtype, device);
    auto work_orgqr_ptr = static_cast<T*>(internal::GetRawOffsetData(work_orgqr));

    device_internals.cusolverdn_handle().Call(
            Orgqr<T>, m, mc, k, q_ptr, lda, tau_ptr, work_orgqr_ptr, buffersize_orgqr, static_cast<int*>(devinfo.get()));

    device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
    if (devinfo_h != 0) {
        throw ChainerxError{"Unsuccessful orgqr (QR) execution. Info = ", devinfo_h};
    }

    q_temp = q_temp.At(std::vector<ArrayIndex>{Slice{0, mc}, Slice{}}).Transpose();  // Q = Q[0:mc, :].T
    r_temp = r_temp.At(std::vector<ArrayIndex>{Slice{}, Slice{0, mc}}).Transpose();  // R = R[:, 0:mc].T
    r_temp = Triu(r_temp, 0);

    device.backend().CallKernel<CopyKernel>(q_temp, q);
    device.backend().CallKernel<CopyKernel>(r_temp, r);
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
    void Call(const Array& a, const Array& u, const Array& s, const Array& vt, bool full_matrices, bool compute_uv) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);

        if (a.shape().GetTotalSize() == 0) {
            if (full_matrices && compute_uv) {
                device.backend().CallKernel<IdentityKernel>(u);
                device.backend().CallKernel<IdentityKernel>(vt);
            }
            // This kernel works correctly for zero-sized input also without early return
            return;
        }

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

        int64_t lda = std::max(int64_t{1}, m);
        int64_t ldu = std::max(int64_t{1}, m);
        int64_t ldvt = full_matrices ? std::max(int64_t{1}, n) : std::max(int64_t{1}, k);

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

            std::shared_ptr<void> devinfo = device.Allocate(sizeof(int));

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
                    lda,
                    s_ptr,
                    vt_ptr,
                    ldu,
                    u_ptr,
                    ldvt,
                    work_ptr,
                    buffersize,
                    nullptr,
                    static_cast<int*>(devinfo.get()));

            int devinfo_h = 0;
            Device& native_device = GetDefaultContext().GetDevice({"native", 0});
            device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
            if (devinfo_h != 0) {
                throw ChainerxError{"Unsuccessful gesvd (SVD) execution. Info = ", devinfo_h};
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

class CudaCholeskyKernel : public CholeskyKernel {
public:
    void Call(const Array& a, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(out.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);
        CHAINERX_ASSERT(out.IsContiguous());
        CHAINERX_ASSERT(a.dtype() == out.dtype());

        // cuSOLVER might not work well with zero-sized arrays for older versions of cuSOLVER (<10.1)
        // therefore it's better to return earlier
        if (a.shape().GetTotalSize() == 0) {
            return;
        }

        // potrf (cholesky) stores result in-place, therefore copy ``a`` to ``out`` and then pass ``out`` to the routine
        device.backend().CallKernel<CopyKernel>(Tril(a, 0), out);

        auto cholesky_impl = [&](auto pt) {
            using T = typename decltype(pt)::type;

            // Note that cuSOLVER uses Fortran order.
            // To compute a lower triangular matrix L = cholesky(A), we use cuSOLVER to compute an upper triangular matrix U = cholesky(A).
            cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            // compute workspace size and prepare workspace
            auto out_ptr = static_cast<T*>(internal::GetRawOffsetData(out));
            int work_size = 0;
            int64_t n = a.shape()[0];
            device_internals.cusolverdn_handle().Call(PotrfBuffersize<T>, uplo, n, out_ptr, std::max(int64_t{1}, n), &work_size);

            // POTRF execution
            Array work = Empty(Shape{work_size}, dtype, device);
            auto work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            std::shared_ptr<void> devinfo = device.Allocate(sizeof(int));
            device_internals.cusolverdn_handle().Call(
                    Potrf<T>, uplo, n, out_ptr, std::max(int64_t{1}, n), work_ptr, work_size, static_cast<int*>(devinfo.get()));

            int devinfo_h = 0;
            Device& native_device = GetDefaultContext().GetDevice({"native", 0});
            device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
            if (devinfo_h != 0) {
                throw ChainerxError{"Unsuccessful potrf (Cholesky) execution. Info = ", devinfo_h};
            }
        };

        VisitFloatingPointDtype(dtype, cholesky_impl);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(CholeskyKernel, CudaCholeskyKernel);

class CudaSyevdKernel : public SyevdKernel {
public:
    void Call(const Array& a, const Array& w, const Array& v, char uplo, bool compute_v) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);

        device.backend().CallKernel<CopyKernel>(a, v);

        int64_t m = a.shape()[0];
        int64_t n = a.shape()[1];

        auto syevd_impl = [&](auto pt) {
            using T = typename decltype(pt)::type;
            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            auto v_ptr = static_cast<T*>(internal::GetRawOffsetData(v));
            auto w_ptr = static_cast<T*>(internal::GetRawOffsetData(w));

            cusolverEigMode_t jobz = compute_v ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

            // cuSOLVER assumes that arrays are stored in column-major order
            // The uplo argument is swapped instead of transposing the input matrix
            cublasFillMode_t uplo_cublas = toupper(uplo) == 'U' ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

            int buffersize = 0;
            // When calling Syevd matrix dimensions are swapped instead of transposing the input matrix
            device_internals.cusolverdn_handle().Call(
                    SyevdBuffersize<T>, jobz, uplo_cublas, n, v_ptr, std::max(int64_t{1}, m), w_ptr, &buffersize);

            Array work = Empty(Shape{buffersize}, dtype, device);
            auto work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            std::shared_ptr<void> devinfo = device.Allocate(sizeof(int));

            device_internals.cusolverdn_handle().Call(
                    Syevd<T>,
                    jobz,
                    uplo_cublas,
                    n,
                    v_ptr,
                    std::max(int64_t{1}, m),
                    w_ptr,
                    work_ptr,
                    buffersize,
                    static_cast<int*>(devinfo.get()));

            int devinfo_h = 0;
            Device& native_device = GetDefaultContext().GetDevice({"native", 0});
            device.MemoryCopyTo(&devinfo_h, devinfo.get(), sizeof(int), native_device);
            if (devinfo_h != 0) {
                throw ChainerxError{"Unsuccessful syevd (Eigen Decomposition) execution. Info = ", devinfo_h};
            }

            // v is stored now in column-major order, need to transform it to row-major
            device.backend().CallKernel<CopyKernel>(v.Transpose(), v);
        };

        VisitFloatingPointDtype(dtype, syevd_impl);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(SyevdKernel, CudaSyevdKernel);

}  // namespace cuda
}  // namespace chainerx
