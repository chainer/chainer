#include "chainerx/routines/loss.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/backend.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/constant.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/math.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {

Array MeanAbsoluteError(const Array& x1, const Array& x2) { return Fabs(x1 - x2).Mean(); }

Array MeanSquaredError(const Array& x1, const Array& x2) { return SquaredDifference(x1, x2).Mean(); }

Array GaussianKLDivergence(const Array& mu, const Array& ln_var) { return (Square(mu) + Exp(ln_var) - ln_var - 1) * 0.5; }

Array HuberLoss(const Array& x1, const Array& x2, Scalar delta) {
    Array a = x1 - x2;
    Array abs_a = Fabs(a);
    Array delta_a = EmptyLike(a);
    delta_a.Fill(delta);

    // TODO(kshitij12345) : use Array < Scalar when implemented.
    return Where(abs_a < delta_a, 0.5 * Square(a), delta * (abs_a - Scalar(0.5) * delta));
}

}  // namespace chainerx
