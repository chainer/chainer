#include "chainerx/routines/evaluation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/backend.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/constant.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/reduction.h"
#include "chainerx/routines/sorting.h"
#include "chainerx/routines/statistics.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {

Array Accuracy(const Array& x1, const Array& x2, const absl::optional<int8_t>& ignore_label) {
    if (ignore_label.has_value()) {
        Array ignore = chainerx::FullLike(x2, Scalar{*ignore_label});
        Array mask = Equal(x2, ignore);
        Array ignore_cnt = Sum(mask);
        Array pred = Where(mask, ignore, ArgMax(x1, 1).Reshape(x2.shape()));
        Array count = Sum(Equal(pred, x2)) - ignore_cnt;
        Scalar size{x2.GetTotalSize()};
        Scalar total = size - AsScalar(ignore_cnt);
        if (total == 0.0) {
            return Array{0}.AsType(x1.dtype());
        } else {
            return Divide(count, total).AsType(x1.dtype());
        }
    } else {
        Array pred = ArgMax(x1, 1).Reshape(x2.shape());
        return Mean(Equal(pred, x2).AsType(x1.dtype()));
    }
}

}  // namespace chainerx
