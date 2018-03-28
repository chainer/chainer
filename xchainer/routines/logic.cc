#include "xchainer/routines/logic.h"

#include "xchainer/array.h"

namespace xchainer {

Array Equal(const Array& x1, const Array& x2) {
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    Array out = Array::Empty(x1.shape(), Dtype::kBool, x1.device());

    x1.device().Equal(x1, x2, out);

    assert(out.shape() == x1.shape());
    assert(out.dtype() == Dtype::kBool);
    assert(out.IsContiguous());
    return out;
}

}  // namespace xchainer
