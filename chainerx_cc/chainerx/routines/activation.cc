#include "chainerx/routines/activation.h"

#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/enum.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"

namespace chainerx {

Array LeakyRelu(const Array& x, Scalar slope) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    Array zero = ZerosLike(x_cast, x_cast.device());
    return Where(x_cast >= zero, x_cast, slope * x_cast);
}

Array ClippedRelu(const Array& x, Scalar z) { return Minimum(Maximum(0, x), z); }

Array Crelu(const Array& x, int8_t axis) {
// TODO(aksub99): Optimize implementation to use a single memory allocation.
    std::vector<Array> c{x, Negative(x)};
    Array concat = Concatenate(c, axis);
    return Relu(concat);
}

Array Sigmoid(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Reciprocal(1 + Exp(-x_cast));
}

Array Relu(const Array& x) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Maximum(0, x_cast);
}

}  // namespace chainerx
