#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

#ifdef CHAINERX_ENABLE_BLAS
#include <cblas.h>
#endif  // CHAINERX_ENABLE_BLAS

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
#include "chainerx/routines/linalg.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace native {

class NativeSVDKernel : public SVDKernel {
public:
    std::tuple<Array, Array, Array> Call(const Array& a, bool full_matrices = true, bool compute_uv = true) override {

        if (a.ndim() != 2) {
            throw DimensionError{"ChainerX SVD decomposition supports only 2-dimensional arrays."};
        }

        throw NotImplementedError("SVD decomposition is not yet implemented for native device");

        if (full_matrices || compute_uv) {
            throw NotImplementedError("SVD decomposition is not yet implemented for native device");;
        }
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(SVDKernel, NativeSVDKernel);

class NativePseudoInverseKernel : public PseudoInverseKernel {
public:
    void Call(const Array& a, const Array& out, float rcond = 1e-15) override {

        if (a.ndim() != 2 || out.ndim() != 2 || rcond != 1.0) {
            throw DimensionError{"ChainerX pseudo-inverse supports only 2-dimensional arrays."};
        }

        throw NotImplementedError("PseudoInverse is not yet implemented for native device");

    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(PseudoInverseKernel, NativePseudoInverseKernel);

}  // namespace native
}  // namespace chainerx
