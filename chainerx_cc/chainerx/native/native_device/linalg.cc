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

class NativeSyevdKernel : public SyevdKernel {
public:
    std::tuple<Array, Array> Call(const Array& a, const std::string& UPLO, bool compute_eigen_vector) override {

        CHAINERX_ASSERT(a.ndim() == 2);

        throw NotImplementedError("Eigen decomposition is not yet implemented for native device");

        if (compute_eigen_vector || UPLO=="L") {
            throw NotImplementedError("Eigen decomposition is not yet implemented for native device");;
        }
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(SyevdKernel, NativeSyevdKernel);

}  // namespace native
}  // namespace chainerx
