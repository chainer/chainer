#include "xchainer/routines/indexing.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <vector>

#include "nonstd/optional.hpp"

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/dtype.h"
#include "xchainer/graph.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/util.h"
#include "xchainer/shape.h"
#include "xchainer/slice.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace internal {
namespace {

// Returns an array where elements at indices are added by the addends `b`.
//
// It is not in-place  operation: the input arrays are not altered.
// It is differentiable with respect to `a` and `b`.
Array AddAt(const Array& a, const std::vector<ArrayIndex>& indices, const Array& b) {
    // TODO(sonots): dtype conversion
    CheckEqual(a.dtype(), b.dtype());

    Array out = a.AsConstant(CopyKind::kCopy);
    Array out_view = out.At(indices);

    // TODO(sonots): broadcasting
    CheckEqual(out_view.shape(), b.shape());

    a.device().Add(b, out_view, out_view);

    auto a_backward_function = [](const Array& gout, const std::vector<GraphId>&) { return gout; };
    auto b_backward_function = [indices](const Array& gout, const std::vector<GraphId>&) { return gout.At(indices); };
    xchainer::internal::SetUpOpNodes("add_at", {a, b}, out, {a_backward_function, b_backward_function});

    return out;
}

}  // namespace

Array At(const Array& a, const std::vector<ArrayIndex>& indices) {
    Shape out_shape;
    Strides out_strides;
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
                assert(false);
        }
    }
    for (int64_t i = i_in; i < a.ndim(); ++i) {
        out_shape.emplace_back(a.shape()[i]);
        out_strides.emplace_back(a.strides()[i]);
    }

    Array out = xchainer::internal::MakeArray(out_shape, out_strides, a.dtype(), a.device(), a.data(), out_offset);

    auto backward_function = [ indices, other = a ](const Array& gout, const std::vector<GraphId>&) {
        Array gin = ZerosLike(other, other.device());
        return AddAt(gin, indices, gout);
    };
    xchainer::internal::SetUpOpNodes("get_item", {a}, out, {backward_function});

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
    assert(0 <= axis && axis < a.ndim());
    assert(b.ndim() == indices.ndim() + a.ndim() - 1);
    CheckEqual(a.dtype(), b.dtype());

    Array out = EmptyLike(a, a.device());

    a.device().AddAt(a, indices, axis, b, out);

    auto a_backward_function = [](const Array& gout, const std::vector<GraphId>&) { return gout; };
    auto b_backward_function = [indices, axis](const Array& gout, const std::vector<GraphId>&) { return Take(gout, indices, axis); };
    xchainer::internal::SetUpOpNodes("add_at", {a, b}, out, {a_backward_function, b_backward_function});

    return out;
}

}  // namespace

Array Take(const Array& a, const Array& indices, int8_t axis) {
    // TODO(niboshi): Support other dtypes by casting
    if (indices.dtype() != Dtype::kInt64) {
        throw DtypeError(
                std::string{"Only "} + GetDtypeName(Dtype::kInt64) + " is supported as indices, but given " +
                GetDtypeName(indices.dtype()));
    }

    int8_t axis_norm = internal::NormalizeAxis(axis, a.ndim());

    Shape out_shape;
    std::copy(a.shape().begin(), a.shape().begin() + axis_norm, std::back_inserter(out_shape));
    std::copy(indices.shape().begin(), indices.shape().end(), std::back_inserter(out_shape));
    std::copy(a.shape().begin() + (axis_norm + 1), a.shape().end(), std::back_inserter(out_shape));
    Array out = Empty(out_shape, a.dtype(), a.device());

    a.device().Take(a, indices, axis_norm, out);

    auto backward_function = [ indices, axis_norm, a_shape = a.shape() ](const Array& gout, const std::vector<GraphId>&) {
        return AddAt(Zeros(a_shape, gout.dtype(), gout.device()), indices, axis_norm, gout);
    };
    xchainer::internal::SetUpOpNodes("take", {a}, out, {backward_function});

    return out;
}

}  // namespace xchainer
