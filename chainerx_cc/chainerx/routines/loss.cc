#include "chainerx/routines/loss.h"

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/misc.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace {

void CheckFloating(Dtype dtype) {
    if (GetKind(dtype) != DtypeKind::kFloat) {
        throw DtypeError{"Loss functions only support floating kind inputs."};
    }
}

void CheckDtypesLossBinary(Dtype dtype1, Dtype dtype2) {
    CheckFloating(dtype1);
    CheckFloating(dtype2);
    CheckEqual(dtype1, dtype2);
}

}  // namespace

Array AbsoluteError(const Array& x1, const Array& x2) {
    CheckDtypesLossBinary(x1.dtype(), x2.dtype());
    return Absolute(x1 - x2);
}

Array SquaredError(const Array& x1, const Array& x2) {
    CheckDtypesLossBinary(x1.dtype(), x2.dtype());
    return Square(x1 - x2);
}

Array GaussianKLDivergence(const Array& mean, const Array& ln_var) {
    CheckDtypesLossBinary(mean.dtype(), ln_var.dtype());
    return (Square(mean) + Exp(ln_var) - ln_var - 1) * 0.5;
}

Array HuberLoss(const Array& x1, const Array& x2, Scalar delta) {
    CheckDtypesLossBinary(x1.dtype(), x2.dtype());
    Array a = x1 - x2;
    Array abs_a = Absolute(a);
    Array delta_array = chainerx::FullLike(a, delta, a.device());

    // TODO(kshitij12345) : use Array < Scalar when implemented.
    return Where(abs_a < delta_array, 0.5 * Square(a), delta * (abs_a - Scalar{0.5} * delta));
}

Array SigmoidCrossEntropy(const Array& x, const Array& t) {
    CheckFloating(x.dtype());
    if (GetKind(t.dtype()) != DtypeKind::kInt) {
        throw DtypeError{"Second argument of SigmoidCrossEntropy should be an array of singed integer dtype."};
    }
    Array ignore_label = -OnesLike(t, t.device());
    Array ignore_mask = NotEqual(t, ignore_label);
    return -(ignore_mask * (x * (t - (GreaterEqual(x, ZerosLike(x, x.device()))).AsType(x.dtype())) - Log1p(Exp(-Absolute(x)))));
}

}  // namespace chainerx
