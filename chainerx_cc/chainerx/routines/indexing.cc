#include "chainerx/routines/indexing.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "absl/types/optional.h"

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/axes.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/constant.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/arithmetic.h"
#include "chainerx/kernels/indexing.h"
#include "chainerx/macro.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/reduction.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/shape.h"
#include "chainerx/slice.h"
#include "chainerx/strides.h"

namespace chainerx {
namespace internal {
namespace {

// Returns an array where elements at indices are added by the addends `b`.
//
// It is not in-place  operation: the input arrays are not altered.
// It is differentiable with respect to `a` and `b`.
Array AtGrad(const Array& a, const std::vector<ArrayIndex>& indices, const Array& b) {
    // TODO(sonots): dtype conversion
    CheckEqual(a.dtype(), b.dtype());

    Array out = a.AsGradStopped(CopyKind::kCopy);
    Array out_view = out.At(indices);

    // TODO(sonots): broadcasting
    CheckEqual(out_view.shape(), b.shape());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<AddKernel>(b, out_view, out_view);
    }

    {
        BackwardBuilder bb{"add_at", {a, b}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([indices](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad()->At(indices); });
        }
        bb.Finalize();
    }

    return out;
}

}  // namespace

Array At(const Array& a, const std::vector<ArrayIndex>& indices) {
    std::vector<ArrayIndex> normalized_indices = internal::GetNormalizedArrayIndices(indices, a.ndim());
    Shape out_shape{};
    Strides out_strides{};
    int64_t out_offset = a.offset();
    bool is_a_empty = a.GetTotalSize() == 0;
    int64_t i_in = 0;
    for (const ArrayIndex& index : normalized_indices) {
        switch (index.tag()) {
            case ArrayIndexTag::kSingleElement: {
                int64_t dim = a.shape()[i_in];
                if (index.index() < -dim || dim <= index.index()) {
                    throw IndexError{"Index ", index.index(), " is out of bounds for axis ", i_in, " with size ", dim};
                }
                if (!is_a_empty) {
                    out_offset += a.strides()[i_in] * ((index.index() + dim) % dim);
                }
                ++i_in;
                break;
            }
            case ArrayIndexTag::kSlice: {
                const Slice& slice = index.slice();
                int64_t slice_length = slice.GetLength(a.shape()[i_in]);
                out_shape.emplace_back(slice_length);
                out_strides.emplace_back(a.strides()[i_in] * slice.step());
                if (!is_a_empty) {
                    int64_t start = slice.GetStart(a.shape()[i_in]);
                    if (start > 0) {
                        out_offset += a.strides()[i_in] * start;
                    }
                }
                ++i_in;
                break;
            }
            case ArrayIndexTag::kNewAxis:
                out_shape.emplace_back(1);
                out_strides.emplace_back(0);
                break;
            default:
                throw ChainerxError{"Invalid ArrayIndexTag."};
        }
    }
    for (int64_t i = i_in; i < a.ndim(); ++i) {
        out_shape.emplace_back(a.shape()[i]);
        out_strides.emplace_back(a.strides()[i]);
    }

    Array out = MakeArray(out_shape, out_strides, a.dtype(), a.device(), a.data(), out_offset);

    BackwardBuilder bb{"get_item", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([indices = std::move(normalized_indices), a_shape = a.shape(), a_dtype = a.dtype()](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            Array gin = Zeros(a_shape, a_dtype, gout.device());
            bctx.input_grad() = AtGrad(gin, indices, gout);
        });
    }
    bb.Finalize();

    return out;
}

}  // namespace internal

Array AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, IndexBoundsMode mode) {
    if (b.ndim() != indices.ndim() + a.ndim() - 1) {
        throw DimensionError{"Input dimensions are invalid. a: ", a.ndim(), ", b:", b.ndim(), ", indices:", indices.ndim(), "."};
    }

    if (!(0 <= axis && axis < a.ndim())) {
        throw DimensionError{"Axis ", axis, " is out of bounds for array of dimension ", a.ndim()};
    }

    CheckEqual(a.dtype(), b.dtype());

    CHAINERX_ASSERT(internal::GetArrayBody(indices)->nodes().empty());

    Array out = EmptyLike(a, a.device());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<AddAtKernel>(a, indices, axis, b, out, mode);
    }

    {
        BackwardBuilder bb{"add_at", {a, b}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            CHAINERX_ASSERT(internal::GetArrayBody(indices)->nodes().empty());
            bt.Define([indices, axis, mode](BackwardContext& bctx) { bctx.input_grad() = Take(*bctx.output_grad(), indices, axis, mode); });
        }
        bb.Finalize();
    }

    return out;
}

Array Take(const Array& a, const Array& indices, int8_t axis, IndexBoundsMode mode) {
    DtypeKind indices_kind = GetKind(indices.dtype());
    if (!(indices_kind == DtypeKind::kInt || indices_kind == DtypeKind::kUInt)) {
        throw DtypeError{"Dtype ", GetDtypeName(indices.dtype()), " cannot be used as an indices array."};
    }

    CHAINERX_ASSERT(internal::GetArrayBody(indices)->nodes().empty());

    int8_t axis_norm = internal::NormalizeAxis(axis, a.ndim());

    Shape out_shape{};
    std::copy(a.shape().begin(), a.shape().begin() + axis_norm, std::back_inserter(out_shape));
    std::copy(indices.shape().begin(), indices.shape().end(), std::back_inserter(out_shape));
    std::copy(a.shape().begin() + (axis_norm + 1), a.shape().end(), std::back_inserter(out_shape));
    Array out = Empty(out_shape, a.dtype(), a.device());

    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<TakeKernel>(a, indices, axis_norm, out, mode);
    }

    BackwardBuilder bb{"take", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        CHAINERX_ASSERT(internal::GetArrayBody(indices)->nodes().empty());
        bt.Define([indices, axis_norm, a_shape = a.shape(), mode](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            // TODO(hvy): Reduce memory allocation for computing the input gradient, i.e. do not allocate a zero-filled array in addition to
            // the output of `AddAt`.
            bctx.input_grad() = AddAt(Zeros(a_shape, gout.dtype(), gout.device()), indices, axis_norm, gout, mode);
        });
    }
    bb.Finalize();

    return out;
}

Array Where(const Array& condition, const Array& x, const Array& y) {
    Dtype out_dtype = ResultType(x, y);
    Shape out_shape = internal::BroadcastShapes(condition.shape(), internal::BroadcastShapes(x.shape(), y.shape()));
    Array out = Empty(out_shape, out_dtype, condition.device());
    Array x_b = x.BroadcastTo(out_shape);
    Array y_b = y.BroadcastTo(out_shape);
    Array condition_b = condition.BroadcastTo(out_shape);

    {
        NoBackpropModeScope scope;
        condition.device().backend().CallKernel<WhereKernel>(condition_b, x_b, y_b, out);
    }

    BackwardBuilder bb{"where", {x_b, y_b}, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([dtype = x.dtype(), condition = condition_b](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& gx = Where(condition, gout, Scalar{0, GetKind(gout.dtype())});
            bctx.input_grad() = gx.dtype() == dtype ? gx : gx.AsType(dtype);
        });
    }
    if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
        bt.Define([dtype = y.dtype(), condition = condition_b](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& gy = Where(condition, Scalar{0, GetKind(gout.dtype())}, gout);
            bctx.input_grad() = gy.dtype() == dtype ? gy : gy.AsType(dtype);
        });
    }
    bb.Finalize();

    return out;
}

Array Where(const Array& condition, const Array& x, Scalar y) {
    Dtype out_dtype = ResultType(x, y);
    Shape out_shape = internal::BroadcastShapes(condition.shape(), x.shape());
    Array out = Empty(out_shape, out_dtype, condition.device());
    Array x_b = x.BroadcastTo(out_shape);
    Array condition_b = condition.BroadcastTo(out_shape);

    {
        NoBackpropModeScope scope;
        condition.device().backend().CallKernel<WhereAASKernel>(condition_b, x_b, y, out);
    }

    BackwardBuilder bb{"where_array_scalar", {x_b}, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([dtype = x.dtype(), condition = std::move(condition_b)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& gx = Where(condition, gout, Scalar{0, GetKind(gout.dtype())});
            bctx.input_grad() = gx.dtype() == dtype ? gx : gx.AsType(dtype);
        });
    }
    bb.Finalize();

    return out;
}

Array Where(const Array& condition, Scalar x, const Array& y) {
    Dtype out_dtype = ResultType(x, y);
    Shape out_shape = internal::BroadcastShapes(condition.shape(), y.shape());
    Array out = Empty(out_shape, out_dtype, condition.device());
    Array y_b = y.BroadcastTo(out_shape);
    Array condition_b = condition.BroadcastTo(out_shape);

    {
        NoBackpropModeScope scope;
        condition.device().backend().CallKernel<WhereASAKernel>(condition_b, x, y_b, out);
    }

    BackwardBuilder bb{"where_scalar_array", {y_b}, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([dtype = y.dtype(), condition = std::move(condition_b)](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            const Array& gy = Where(condition, Scalar{0, GetKind(gout.dtype())}, gout);
            bctx.input_grad() = gy.dtype() == dtype ? gy : gy.AsType(dtype);
        });
    }
    bb.Finalize();

    return out;
}

Array Where(const Array& condition, Scalar x, Scalar y) {
    Dtype out_dtype = ResultType(x, y);
    Array out = Empty(condition.shape(), out_dtype, condition.device());
    {
        NoBackpropModeScope scope;
        condition.device().backend().CallKernel<WhereASSKernel>(condition, x, y, out);
    }
    return out;
}

std::vector<Array> Nonzero(const Array& a) {
    if (a.ndim() == 0) {
        throw DimensionError{"0-dim inputs not allowed."};
    }

    NoBackpropModeScope scope{};
    Array is_nonzero = (a != ZerosLike(a)).Ravel();
    int64_t count_nonzero = static_cast<int64_t>(AsScalar(is_nonzero.Sum()));
    Array raw_index = Zeros(Shape{count_nonzero}, Dtype::kInt64, a.device());
    if (count_nonzero > 0) {
        Array addat_indices = Where(is_nonzero, Maximum(Cumsum(is_nonzero) - 1, 0), 0);
        Array indices = Where(is_nonzero, Arange(a.GetTotalSize(), Dtype::kInt64), 0);
        a.device().backend().CallKernel<AddAtKernel>(
                raw_index, std::move(addat_indices), 0, std::move(indices), raw_index, IndexBoundsMode::kDefault);
    }

    std::vector<Array> out;
    out.reserve(a.ndim());
    for (int8_t i = 0; i < a.ndim(); ++i) {
        int64_t step = Shape{a.shape().begin() + i + 1, a.shape().end()}.GetTotalSize();
        out.emplace_back(FloorDivide(raw_index, step) % a.shape()[i]);
    }
    return out;
}

}  // namespace chainerx
