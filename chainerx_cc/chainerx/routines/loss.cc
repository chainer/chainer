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
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {

Array MeanAbsoluteError(const Array& x0, const Array& x1) { return Fabs(x0 - x1).Mean(); }

Array MeanSquaredError(const Array& x0, const Array& x1) { return SquaredDifference(x0, x1).Mean(); }

Array GaussianKLDivergence(const Array& mu, const Array& ln_var, const std::string& reduce) {
    const Array& var = Exp(ln_var);
    const Array& mean_square = Square(mu);

    Array loss = (mean_square + var - ln_var - 1) * 0.5;

    if (reduce == "sum") {
        return loss.Sum();
    } else if (reduce == "mean") {
        return loss.Mean();
    } else {
        return loss;
    }
}

Array HuberLoss(const Array& x, const Array& t, Scalar delta, const std::string& reduce) {
    Array a = x - t;
    Array abs_a = Fabs(a);
    Array delta_a = EmptyLike(a);
    delta_a.Fill(delta);

    // TODO(kshitij12345) : use Array < Scalar when implemented.
    Array loss = Where(abs_a < delta_a, 0.5 * Square(a), delta * (abs_a - Scalar(0.5) * delta));
    if (reduce == "sum_along_second_axis") {
        return loss.Sum(Axes{1});
    }

    return loss;
}

}  // namespace chainerx
