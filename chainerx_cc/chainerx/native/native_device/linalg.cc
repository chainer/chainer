#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "chainerx/array.h"
#include "chainerx/backend.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"

#if CHAINERX_ENABLE_LAPACK
extern "C" {
// geqrf
void dgeqrf_(int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
void sgeqrf_(int* m, int* n, float* a, int* lda, float* tau, float* work, int* lwork, int* info);

// orgqr
void dorgqr_(int* m, int* n, int* k, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
void sorgqr_(int* m, int* n, int* k, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
}
#endif  // CHAINERX_ENABLE_LAPACK

namespace chainerx {
namespace native {
namespace {

template <typename T>
void Geqrf(int /*m*/, int /*n*/, T* /*a*/, int /*lda*/, T* /*tau*/, T* /*work*/, int /*lwork*/, int* /*info*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by geqrf (QR)"};
}

template <typename T>
void Orgqr(int /*m*/, int /*n*/, int /*k*/, T* /*a*/, int /*lda*/, T* /*tau*/, T* /*work*/, int /*lwork*/, int* /*info*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by orgqr (QR)"};
}

#if CHAINERX_ENABLE_LAPACK
template <>
void Geqrf<double>(int m, int n, double* a, int lda, double* tau, double* work, int lwork, int* info) {
    dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template <>
void Geqrf<float>(int m, int n, float* a, int lda, float* tau, float* work, int lwork, int* info) {
    sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
}

template <>
void Orgqr<double>(int m, int n, int k, double* a, int lda, double* tau, double* work, int lwork, int* info) {
    dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}

template <>
void Orgqr<float>(int m, int n, int k, float* a, int lda, float* tau, float* work, int lwork, int* info) {
    sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
}
#endif  // CHAINERX_ENABLE_LAPACK

template <typename T>
std::tuple<Array, Array> QRImpl(const Array& a, QRMode mode) {
    Device& device = a.device();
    Dtype dtype = a.dtype();

    int m = a.shape()[0];
    int n = a.shape()[1];
    int mn = std::min(m, n);

    Array Q = Empty(Shape({0}), dtype, device);
    Array R = a.Transpose().Copy();  // QR decomposition is done in-place
    Array tau = Empty(Shape({mn}), dtype, device);

    T* r_ptr = static_cast<T*>(internal::GetRawOffsetData(R));
    T* tau_ptr = static_cast<T*>(internal::GetRawOffsetData(tau));

    int info;
    int buffersize_geqrf = -1;
    T work_query_geqrf;
    Geqrf(m, n, r_ptr, m, tau_ptr, &work_query_geqrf, buffersize_geqrf, &info);
    buffersize_geqrf = static_cast<int>(work_query_geqrf);

    Array work = Empty(Shape({buffersize_geqrf}), dtype, device);
    T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

    Geqrf(m, n, r_ptr, m, tau_ptr, work_ptr, buffersize_geqrf, &info);

    if (info != 0) {
        throw ChainerxError{"Unsuccessful geqrf (QR) execution. Info = ", info};
    }

    if (mode == QRMode::r) {
        R = R.At(std::vector<ArrayIndex>{Slice{}, Slice{0, mn}}).Transpose();  // R = R[:, 0:mn].T
        R = Triu(R, 0);
        return std::make_tuple(std::move(Q), std::move(R));
    }

    if (mode == QRMode::raw) {
        return std::make_tuple(std::move(R), std::move(tau));
    }

    int mc;
    if (mode == QRMode::complete && m > n) {
        mc = m;
        Q = Empty(Shape({m, m}), dtype, device);
    } else {
        mc = mn;
        Q = Empty(Shape({n, m}), dtype, device);
    }

    device.backend().CallKernel<CopyKernel>(R, Q.At(std::vector<ArrayIndex>{Slice{0, n}, Slice{}}));  // Q[0:n, :] = R
    T* q_ptr = static_cast<T*>(internal::GetRawOffsetData(Q));

    int buffersize_orgqr = -1;
    T work_query_orgqr;
    Orgqr(m, mc, mn, q_ptr, m, tau_ptr, &work_query_orgqr, buffersize_orgqr, &info);
    buffersize_orgqr = static_cast<int>(work_query_orgqr);

    Array work_orgqr = Empty(Shape({buffersize_orgqr}), dtype, device);
    T* work_orgqr_ptr = static_cast<T*>(internal::GetRawOffsetData(work_orgqr));

    Orgqr(m, mc, mn, q_ptr, m, tau_ptr, work_orgqr_ptr, buffersize_orgqr, &info);

    if (info != 0) {
        throw ChainerxError{"Unsuccessful orgqr (QR) execution. Info = ", info};
    }

    // .Copy() is needed to have correct strides
    Q = Q.At(std::vector<ArrayIndex>{Slice{0, mc}, Slice{}}).Transpose().Copy();  // Q = Q[0:mc, :].T
    R = R.At(std::vector<ArrayIndex>{Slice{}, Slice{0, mc}}).Transpose();  // R = R[:, 0:mc].T
    R = Triu(R, 0);
    return std::make_tuple(std::move(Q), std::move(R));
}

}  // namespace

class NativeQRKernel : public QRKernel {
public:
    std::tuple<Array, Array> Call(const Array& a, QRMode mode = QRMode::reduced) override {
#if CHAINERX_ENABLE_LAPACK
        CHAINERX_ASSERT(a.ndim() == 2);

        return VisitFloatingPointDtype(a.dtype(), [&](auto pt) -> std::tuple<Array, Array> {
            using T = typename decltype(pt)::type;
            return QRImpl<T>(a, mode);
        });
#else  // CHAINERX_ENABLE_LAPACK
        (void)a;  // unused
        (void)mode;  // unused
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_ENABLE_LAPACK
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(QRKernel, NativeQRKernel);

}  // namespace native
}  // namespace chainerx
