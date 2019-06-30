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
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/reduction.h"
#include "chainerx/routines/statistics.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {

Array Accuracy(const Array& x1, const Array& x2, const nonstd::optional<Array>& ignore_label) {
    if (ignore_label.has_value()) {
        Array mask = Equal(x2, *ignore_label);
        Array ignore_cnt = Sum(mask);
        Array pred = Where(mask, *ignore_label, AMax(x1, 1).Reshape(x2.shape()));
        Array count = Sum(Equal(pred, x2)) - ignore_cnt;
        Scalar size{x2.GetTotalSize()};
        Scalar total = size - AsScalar(ignore_cnt);
        if (total == 0.0) {
            return Array{0};
        } else {
            return count / total;
        }
    } else {
        Array pred = AMax(x2, 1).Reshape(x2.shape());
        return Mean(Equal(pred, x2));
    }
}

}  // namespace chainerx
