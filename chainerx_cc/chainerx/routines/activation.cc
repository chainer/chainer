#include "chainerx/routines/activation.h"

#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/device.h"
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

Array ClippedRelu(const Array& x, Scalar z) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    return Minimum(Maximum(0, x_cast), z);
}

Array CRelu(const Array& x, int8_t axis) {
    // TODO(aksub99): Optimize implementation to use a single memory allocation.
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    std::vector<Array> c{x_cast, Negative(x_cast)};
    Array concat = Concatenate(c, axis);
    return Relu(concat);
}

Array Elu(const Array& x, double alpha) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    // TODO(aksub99): Replace x > zero with x > 0 when operator > supports scalars.
    Array zero = Zeros({}, x_cast.dtype(), x_cast.device());
    return Where(x_cast > zero, x_cast, alpha * Expm1(x_cast));
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

Array LeakyRelu(const Array& x, Scalar slope) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    // TODO(hamaji): Replace x >= zero with x >= 0 when operator >= supports scalars.
    Array zero = Zeros({}, x_cast.dtype(), x_cast.device());
    return Where(x_cast >= zero, x_cast, slope * x_cast);
}

Array Softplus(const Array& x, double beta) {
    Dtype dtype = internal::GetMathResultDtype(x.dtype());
    const Array& x_cast = x.dtype() == dtype ? x : x.AsType(dtype);
    double beta_inv = 1.0 / beta;
    Array bx = beta * x_cast;
    Array y = (Maximum(bx, 0) + Log1p(Exp(-Fabs(bx)))) * beta_inv;
    return y;
}

}  // namespace chainerx
