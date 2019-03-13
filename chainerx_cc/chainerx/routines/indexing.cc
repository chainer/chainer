#include "chainerx/routines/indexing.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "nonstd/optional.hpp"

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/axes.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/constant.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
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
Array AddAt(const Array& a, const std::vector<ArrayIndex>& indices, const Array& b) {
    // TODO(sonots): dtype conversion
    CheckEqual(a.dtype(), b.dtype());

    Array out = a.AsGradStopped(CopyKind::kCopy);
    Array out_view = out.At(indices);

    // TODO(sonots): broadcasting
    CheckEqual(out_view.shape(), b.shape());

    {
        NoBackpropModeScope scope{};
        a.device().Add(b, out_view, out_view);
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
    Shape out_shape{};
    Strides out_strides{};
    int64_t out_offset = a.offset();
    int64_t i_in = 0;
    for (const ArrayIndex& index : indices) {
        switch (index.tag()) {
            case ArrayIndexTag::kSingleElement: {
                int64_t dim = a.shape()[i_in];
                if (index.index() < -dim || dim <= index.index()) {
                    throw DimensionError{"Index ", index.index(), " is out of bounds for axis ", i_in, " with size ", dim};
                }
                out_offset += a.strides()[i_in] * ((index.index() + dim) % dim);
                ++i_in;
                break;
            }
            case ArrayIndexTag::kSlice: {
                const Slice& slice = index.slice();
                int64_t slice_length = slice.GetLength(a.shape()[i_in]);
                out_offset += a.strides()[i_in] * slice.GetStart(a.shape()[i_in]);
                out_shape.emplace_back(slice_length);
                out_strides.emplace_back(a.strides()[i_in] * slice.step());
                ++i_in;
                break;
            }
            case ArrayIndexTag::kNewAxis:
                out_shape.emplace_back(1);
                out_strides.emplace_back(0);
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    }
    for (int64_t i = i_in; i < a.ndim(); ++i) {
        out_shape.emplace_back(a.shape()[i]);
        out_strides.emplace_back(a.strides()[i]);
    }

    Array out = MakeArray(out_shape, out_strides, a.dtype(), a.device(), a.data(), out_offset);

    BackwardBuilder bb{"get_item", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([indices, a_shape = a.shape(), a_dtype = a.dtype()](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            Array gin = Zeros(a_shape, a_dtype, gout.device());
            bctx.input_grad() = AddAt(gin, indices, gout);
        });
    }
    bb.Finalize();

    return out;
}

}  // namespace internal

namespace {

// Adds elements of `b` indexed by `indices` into `a` and returns the result.
// Used in backward pass of Take()
//
// It is not in-place operation: the input arrays are not altered.
// It is differentiable with respect to `a` and `b`.
Array AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b) {
    CHAINERX_ASSERT(0 <= axis && axis < a.ndim());
    CHAINERX_ASSERT(b.ndim() == indices.ndim() + a.ndim() - 1);
    CheckEqual(a.dtype(), b.dtype());

    CHAINERX_ASSERT(internal::GetArrayBody(indices)->nodes().empty());

    Array out = EmptyLike(a, a.device());

    {
        NoBackpropModeScope scope{};
        a.device().AddAt(a, indices, axis, b, out);
    }

    {
        BackwardBuilder bb{"add_at", {a, b}, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
        }
        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            CHAINERX_ASSERT(internal::GetArrayBody(indices)->nodes().empty());
            bt.Define([indices, axis](BackwardContext& bctx) { bctx.input_grad() = Take(*bctx.output_grad(), indices, axis); });
        }
        bb.Finalize();
    }

    return out;
}

}  // namespace

Array Take(const Array& a, const Array& indices, int8_t axis) {
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
        a.device().Take(a, indices, axis_norm, out);
    }

    BackwardBuilder bb{"take", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        CHAINERX_ASSERT(internal::GetArrayBody(indices)->nodes().empty());
        bt.Define([indices, axis_norm, a_shape = a.shape()](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = AddAt(Zeros(a_shape, gout.dtype(), gout.device()), indices, axis_norm, gout);
        });
    }
    bb.Finalize();

    return out;
}

}  // namespace chainerx
