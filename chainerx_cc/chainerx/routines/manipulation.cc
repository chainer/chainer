#include "chainerx/routines/manipulation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/device.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {

Scalar AsScalar(const Array& a) {
    if (a.GetTotalSize() != 1) {
        throw DimensionError{"Cannot convert an array of size ", a.GetTotalSize(), " to a scalar, size must be 1."};
    }

    // Copy to the native device
    Array native_copy = a.ToNative();

    // Retrieve the value
    return VisitDtype(a.dtype(), [&native_copy](auto pt) -> Scalar {
        using T = typename decltype(pt)::type;
        const uint8_t* ptr = static_cast<const uint8_t*>(native_copy.data().get()) + native_copy.offset();
        auto typed_ptr = reinterpret_cast<const T*>(ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        return Scalar{*typed_ptr};
    });
}

Array RollAxis(const Array& a, int8_t axis, int8_t start) {
    // TODO(hvy): Optimize the implementation.
    axis = internal::NormalizeAxis(axis, a.ndim());

    // start can be a.ndim() so we cannot use NormalizeAxis here.
    if (start < -a.ndim() || start > a.ndim()) {
        throw DimensionError{"start arg out of bounds. start: ", start, ", ndim: ", a.ndim()};
    }
    if (start < 0) {
        start += a.ndim();
    }

    Axes axes;
    for (int8_t i = 0; i < a.ndim(); ++i) {
        if (i == start) {
            axes.emplace_back(axis);
        }
        if (i != axis) {
            axes.emplace_back(i);
        }
    }
    if (start == a.ndim()) {
        axes.emplace_back(axis);
    }
    return Transpose(a, axes);
}

Array Transpose(const Array& a, const OptionalAxes& axes) {
    Axes real_axes;
    if (axes.has_value()) {
        if (axes->ndim() != a.ndim()) {
            throw DimensionError{"Axes do not match, input array dimensions: ", a.ndim(), " but axes: ", axes->ndim()};
        }
        real_axes = internal::GetNormalizedAxes(*axes, a.ndim());
    } else {
        for (int8_t i = 0; i < a.ndim(); ++i) {
            real_axes.emplace_back(a.ndim() - i - 1);
        }
    }
    CHAINERX_ASSERT(real_axes.ndim() == a.ndim());

    Shape out_shape;
    Strides out_strides;
    for (int8_t axis : real_axes) {
        out_shape.emplace_back(a.shape()[axis]);
        out_strides.emplace_back(a.strides()[axis]);
    }

    Array out = internal::MakeArray(out_shape, out_strides, a.dtype(), a.device(), a.data(), a.offset());

    BackwardBuilder bb{"transpose", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([real_axes](BackwardContext& bctx) {
            Axes backward_axes;
            backward_axes.resize(real_axes.ndim());
            for (int8_t i = 0; i < real_axes.ndim(); ++i) {
                backward_axes[real_axes[i]] = i;
            }
            bctx.input_grad() = bctx.output_grad()->Transpose(backward_axes);
        });
    }
    bb.Finalize();

    return out;
}

namespace {

// Returns a shape where the length of at most one dimension is inferred from the total size and the remaining dimensions.
// Such a dimension is given by a negative length, i.e. Shape{2, 3, -1}.
// If the given shape does not contain such a dimension, this function will return a copy of the given shape.
// If there exists multiple negative lengths or if the negative length dimension cannot be inferred due to non divisbility, an
// DimensionError is thrown.
Shape GetInferredShape(const Shape& shape, int64_t total_size) {
    Shape inferred_shape = shape;

    auto it = std::find_if(inferred_shape.begin(), inferred_shape.end(), [](int64_t dim) { return dim < 0; });
    if (it != inferred_shape.end()) {
        if (std::find_if(std::next(it), inferred_shape.end(), [](int64_t dim) { return dim < 0; }) != inferred_shape.end()) {
            throw DimensionError{"Can only specify one unknown dimension"};
        }
        int64_t rest_size = std::accumulate(inferred_shape.begin(), it, int64_t{1}, std::multiplies<>()) *
                            std::accumulate(std::next(it), inferred_shape.end(), int64_t{1}, std::multiplies<>());
        *it = total_size / rest_size;
    }

    if (total_size != inferred_shape.GetTotalSize()) {
        throw DimensionError{"Cannot reshape array of size ", total_size, " into shape ", shape};
    }
    return inferred_shape;
}

}  // namespace

Array Reshape(const Array& a, const Shape& newshape) {
    const Shape& in_shape = a.shape();
    const Strides& in_strides = a.strides();

    // If the shape is unchanged, just return a view.
    if (in_shape == newshape) {
        return a.MakeView();
    }

    // Check for invalid shape.
    int64_t total_size = in_shape.GetTotalSize();
    Shape out_shape = GetInferredShape(newshape, total_size);
    int64_t item_size = a.GetItemSize();
    Strides strides{};
    if (total_size == 0) {
        // Calculate the strides for 0-sized array.
        strides.resize(out_shape.ndim());
        strides.back() = item_size;
        for (int8_t i = out_shape.ndim() - 1; i >= 1; --i) {
            strides[i - 1] = strides[i] * std::max(int64_t{1}, out_shape[i]);
        }
    } else {
        // Calculate the strides for non-0-sized array.

        // reduced_shape and reduced_strides are the shortest shape and strides which can be convertible from input shape and strides
        // without copy.
        Shape reduced_shape{};
        Strides reduced_strides{};
        if (total_size == 1) {
            reduced_shape.push_back(int64_t{1});
            reduced_strides.push_back(item_size);
        } else {
            int8_t i = 0;
            // Ignore preceding 1-length dimensions
            while (i < in_shape.ndim() && in_shape[i] == 1) {
                ++i;
            }
            // Add the first pair
            reduced_shape.emplace_back(in_shape[i]);
            reduced_strides.emplace_back(in_strides[i]);
            ++i;
            // Reduce the remaining
            for (; i < in_shape.ndim(); ++i) {
                int64_t dim = in_shape[i];
                int64_t st = in_strides[i];
                CHAINERX_ASSERT(dim > 0);
                if (dim == 1 && st == 0) {
                    // If the axis has unit-length with no stride, skip this dimension.
                } else if (dim * st == reduced_strides.back()) {
                    // If the pair is compatible with the previous stride, reduce the pair to it.
                    reduced_shape.back() *= dim;
                    reduced_strides.back() = st;
                } else {
                    // Otherwise, add a new shape and stride.
                    reduced_shape.push_back(dim);
                    reduced_strides.push_back(st);
                }
            }
        }
        CHAINERX_ASSERT(reduced_shape.size() == reduced_strides.size());
        CHAINERX_ASSERT(!reduced_shape.empty());

        // Construct the strides for no-copy reshape.
        // If it's not possible, can_reshape_without_copy will be false.
        bool can_reshape_without_copy = true;
        if (out_shape.ndim() > 0) {
            int64_t last_stride = reduced_shape[0] * reduced_strides[0];
            size_t i_dim = 0;
            for (int64_t dim : out_shape) {
                if (dim <= 1) {
                    strides.push_back(last_stride);
                    continue;
                }
                if (i_dim >= reduced_shape.size() || reduced_shape[i_dim] % dim != 0) {
                    strides.clear();
                    can_reshape_without_copy = false;
                    break;
                }
                reduced_shape[i_dim] /= dim;
                last_stride = reduced_shape[i_dim] * reduced_strides[i_dim];
                strides.push_back(last_stride);
                if (reduced_shape[i_dim] == 1) {
                    ++i_dim;
                }
            }
        }

        if (!can_reshape_without_copy) {
            // Copy is required.
            return a.Copy().Reshape(out_shape);
        }
        CHAINERX_ASSERT(strides.size() == out_shape.size());
    }

    Array out = internal::MakeArray(out_shape, strides, a.dtype(), a.device(), a.data(), a.offset());

    BackwardBuilder bb{"reshape", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([in_shape](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad()->Reshape(in_shape); });
    }
    bb.Finalize();

    CHAINERX_ASSERT(out.shape() == out_shape);
    CHAINERX_ASSERT(out.strides().size() == out_shape.size());
    return out;
}

Array Squeeze(const Array& a, const OptionalAxes& axis) {
    const Shape& in_shape = a.shape();
    const Strides& in_strides = a.strides();

    Shape out_shape{};
    Strides out_strides{};

    if (axis.has_value()) {
        const Axes sorted_axis = internal::GetSortedAxes(*axis, in_shape.ndim());

        int64_t i_axis = 0;
        for (int64_t i = 0; i < in_shape.ndim(); ++i) {
            if (i_axis < static_cast<int64_t>(sorted_axis.size()) && sorted_axis[i_axis] == i) {
                ++i_axis;
                if (in_shape[i] != 1) {
                    std::ostringstream os;
                    os << "Cannot squeeze out non-unit-length axes, where shape was " << in_shape.ToString();
                    os << " and axes were (";
                    for (auto it = axis->begin(); it != axis->end(); ++it) {
                        if (it != axis->begin()) {
                            os << ", ";
                        }
                        os << *it;
                    }
                    os << (axis->size() == 1 ? ",)." : ").");
                    throw DimensionError{os.str()};
                }
            } else {
                out_shape.emplace_back(in_shape[i]);
                out_strides.emplace_back(in_strides[i]);
            }
        }
    } else {  // All axes are candidates for removal if none are given.
        for (int64_t i = 0; i < in_shape.ndim(); ++i) {
            if (in_shape[i] != 1) {
                out_shape.emplace_back(in_shape[i]);
                out_strides.emplace_back(in_strides[i]);
            }
        }
    }

    if (in_shape.size() == out_shape.size()) {
        return a;
    }

    Array out = internal::MakeArray(out_shape, out_strides, a.dtype(), a.device(), a.data(), a.offset());

    BackwardBuilder bb{"squeeze", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([in_shape](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad()->Reshape(in_shape); });
    }
    bb.Finalize();

    return out;
}

Array BroadcastTo(const Array& array, const Shape& shape) {
    const Shape& in_shape = array.shape();
    const Strides& in_strides = array.strides();

    if (in_shape.size() > shape.size()) {
        throw DimensionError{"Cannot broadcast to smaller dimensions from ", in_shape, " to ", shape, "."};
    }

    // Compute the new set of strides after broadcastining.
    Strides strides;
    strides.resize(shape.ndim());
    int8_t i_in = in_shape.ndim() - 1;
    for (int8_t i_out = shape.ndim() - 1; i_out >= 0; --i_out) {
        int64_t out_dim = shape[i_out];
        // If this dimension is to be broadcasted, nonbroadcast_stride is unset.
        // Otherwise, it holds the new stride.
        nonstd::optional<int64_t> nonbroadcast_stride{};
        if (i_in >= 0) {
            int64_t in_dim = in_shape[i_in];
            if (in_dim == 1) {
                // do nothing; broadcast
            } else if (in_dim == out_dim) {
                nonbroadcast_stride = in_strides[i_in];
            } else {
                throw DimensionError{"Invalid broadcast from ", in_shape, " to ", shape};
            }
            --i_in;
        } else {
            // do nothing; broadcast
        }

        if (nonbroadcast_stride.has_value()) {
            // non-broadcast dimension
            strides[i_out] = nonbroadcast_stride.value();
        } else {
            // broadcast dimension
            strides[i_out] = int64_t{0};
        }
    }
    CHAINERX_ASSERT(i_in == -1);
    CHAINERX_ASSERT(strides.ndim() == shape.ndim());

    Array out = internal::MakeArray(shape, strides, array.dtype(), array.device(), array.data(), array.offset());

    BackwardBuilder bb{"broadcast_to", array, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([in_shape](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            if (gout.shape() == in_shape) {
                bctx.input_grad() = gout;
                return;
            }

            int8_t lead = gout.ndim() - in_shape.ndim();
            Axes lead_axis{};
            lead_axis.resize(lead);
            std::iota(lead_axis.begin(), lead_axis.end(), int8_t{0});

            Axes axis{lead_axis};
            for (int8_t i = 0; i < in_shape.ndim(); ++i) {
                if (in_shape[i] == 1) {
                    axis.emplace_back(i + lead);
                }
            }
            axis.erase(std::unique(axis.begin(), axis.end()), axis.end());  // Sum does not accept axis with duplicate elements

            Array gin = gout.Sum(axis, true);
            if (lead > 0) {
                bctx.input_grad() = gin.Squeeze(lead_axis);
            } else {
                bctx.input_grad() = std::move(gin);
            }
        });
    }
    bb.Finalize();

    return out;
}

std::vector<Array> Split(const Array& ary, int64_t sections, int8_t axis) {
    if (sections < 1) {
        throw DimensionError("Number of sections must be larger than 0.");
    }

    const Shape& in_shape = ary.shape();
    int64_t in_dim = in_shape[axis];
    if (in_dim % sections != 0) {
        throw DimensionError("Array split does not result in an equal division.");
    }

    std::vector<Array> out;

    Shape out_shape = in_shape;
    int64_t& out_dim = out_shape[axis];
    int64_t out_stride = ary.strides()[axis];
    int64_t out_offset = ary.offset();
    int64_t slice_start = 0;
    int64_t slice_step = in_dim / sections;

    for (int64_t i = 0; i < sections; ++i) {
        int64_t slice_stop = std::min(in_dim, slice_start + slice_step);

        // Update the dimension of interest in the output shape.
        out_dim = slice_stop - slice_start;

        out.emplace_back(internal::MakeArray(out_shape, ary.strides(), ary.dtype(), ary.device(), ary.data(), out_offset));

        out_offset += out_stride * out_dim;
        slice_start = slice_stop;
    }
    return out;
}

std::vector<Array> Split(const Array& ary, std::vector<int64_t> indices, int8_t axis) {
    const Shape& in_shape = ary.shape();
    int64_t in_dim = in_shape[axis];

    // Wrap negative indices.
    std::transform(
            indices.begin(), indices.end(), indices.begin(), [in_dim](int64_t index) { return index >= 0 ? index : index + in_dim; });
    indices.emplace_back(in_dim);

    std::vector<Array> out;

    Shape out_shape = in_shape;
    int64_t& out_dim = out_shape[axis];
    int64_t out_stride = ary.strides()[axis];
    int64_t out_offset = ary.offset();
    int64_t slice_start = 0;

    for (int64_t index : indices) {
        int64_t slice_stop = std::min(in_dim, index);
        int64_t slice_step = slice_stop - slice_start;

        // Update the dimension of interest in the output shape.
        out_dim = std::max(int64_t{0}, slice_step);

        out.emplace_back(internal::MakeArray(out_shape, ary.strides(), ary.dtype(), ary.device(), ary.data(), out_offset));

        out_offset += out_stride * slice_step;
        slice_start = slice_stop;
    }
    return out;
}

}  // namespace chainerx
