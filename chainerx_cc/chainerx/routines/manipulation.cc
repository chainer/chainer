#include "chainerx/routines/manipulation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/indexing.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/type_util.h"
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
        if (rest_size == 0) {
            throw DimensionError{"Cannot reshape array of size ", total_size, " into an ambiguous shape ", shape};
        }
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
            reduced_shape.emplace_back(int64_t{1});
            reduced_strides.emplace_back(item_size);
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
                if (dim == 1) {
                    // If the axis has unit-length, skip this dimension.
                } else if (dim * st == reduced_strides.back()) {
                    // If the pair is compatible with the previous stride, reduce the pair to it.
                    reduced_shape.back() *= dim;
                    reduced_strides.back() = st;
                } else {
                    // Otherwise, add a new shape and stride.
                    reduced_shape.emplace_back(dim);
                    reduced_strides.emplace_back(st);
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
                    strides.emplace_back(last_stride);
                    continue;
                }
                if (i_dim >= reduced_shape.size() || reduced_shape[i_dim] % dim != 0) {
                    strides.clear();
                    can_reshape_without_copy = false;
                    break;
                }
                reduced_shape[i_dim] /= dim;
                last_stride = reduced_shape[i_dim] * reduced_strides[i_dim];
                strides.emplace_back(last_stride);
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
        absl::optional<int64_t> nonbroadcast_stride{};
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

namespace {

Array ConcatenateImpl(const std::vector<Array>& arrays, int8_t axis) {
    if (arrays.empty()) {
        throw DimensionError{"Need at least one array to concatenate"};
    }

    Shape shape = arrays.front().shape();
    Dtype out_dtype = ResultType(arrays);
    Device& device = arrays.front().device();
    int8_t ndim = arrays.front().ndim();
    axis = internal::NormalizeAxis(axis, ndim);
    shape[axis] = 0;
    std::vector<int64_t> indices;
    indices.reserve(arrays.size() - 1);

    for (const Array& array : arrays) {
        const Shape& s = array.shape();
        if (ndim != array.ndim()) {
            throw DimensionError{"All the input arrays must have same number of dimensions"};
        }
        for (int8_t i = 0; i < ndim; ++i) {
            if (axis == i) {
                shape[i] += s[i];
            } else if (shape[i] != s[i]) {
                throw DimensionError{"All the input array dimensions except for the concatenation axis must match exactly"};
            }
        }
        if (indices.size() < arrays.size() - 1) {
            indices.emplace_back(shape[axis]);
        }
    }

    Strides strides{shape, out_dtype};

    // Aligning with NumPy strides behavior
    auto last_zero_it = std::find(shape.rbegin(), shape.rend(), int64_t{0});
    if (last_zero_it != shape.rend()) {
        std::fill(strides.rbegin() + (last_zero_it - shape.rbegin() + 1), strides.rend(), int64_t{0});
    }

    Array out = internal::Empty(shape, out_dtype, strides, device);

    size_t in_size = arrays.size();

    // If input dtypes are mixed, elements in the input arrays are casted to the resulting dtype.
    // Their original dtypes must therefore be remembered in order to cast the computed gradients back in the backward pass.
    std::vector<Dtype> in_dtypes;
    in_dtypes.reserve(in_size);

    std::vector<ConstArrayRef> array_refs;
    array_refs.reserve(in_size);

    {
        NoBackpropModeScope scope{};
        int64_t out_offset = 0;
        for (const Array& array : arrays) {
            const Shape& shape = array.shape();
            Array sliced_out = internal::MakeArray(shape, strides, out_dtype, device, out.data(), out_offset);
            Dtype in_dtype = array.dtype();
            in_dtypes.emplace_back(in_dtype);
            // Note: In CopyKernel, Input Array Elements are casted to the type of Output Array.
            device.backend().CallKernel<CopyKernel>(array, sliced_out);
            array_refs.emplace_back(ConstArrayRef{array});
            out_offset += strides[axis] * shape[axis];
        }
    }

    {
        BackwardBuilder bb{"concatenate", array_refs, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget()) {
            bt.Define([indices = std::move(indices), axis, in_dtypes = std::move(in_dtypes)](BackwardContext& bctx) {
                const Array& gy = *bctx.output_grad();
                Dtype out_dtype = gy.dtype();
                std::vector<Array> gxs = Split(gy, indices, axis);
                for (size_t i = 0; i < gxs.size(); ++i) {
                    Dtype in_dtype = in_dtypes[i];
                    if (out_dtype != in_dtype) {
                        bctx.input_grad(i) = gxs[i].AsType(in_dtype);
                    } else {
                        bctx.input_grad(i) = std::move(gxs[i]);
                    }
                }
            });
        }
        bb.Finalize();
    }

    return out;
}

}  // namespace

Array Concatenate(const std::vector<Array>& arrays) { return ConcatenateImpl(arrays, 0); }

Array Concatenate(const std::vector<Array>& arrays, absl::optional<int8_t> axis) {
    if (!axis.has_value()) {
        // Special case, making input arrays 1-dimensional and concatenating along the first axis.
        std::vector<Array> raveled_arrays;
        raveled_arrays.reserve(arrays.size());
        std::transform(arrays.begin(), arrays.end(), std::back_inserter(raveled_arrays), [](const Array& array) {
            Shape shape{array.GetTotalSize()};
            return array.Reshape(shape);
        });
        return ConcatenateImpl(raveled_arrays, 0);
    }
    return ConcatenateImpl(arrays, *axis);
}

Array Stack(const std::vector<Array>& arrays, int8_t axis) {
    std::vector<Array> reshaped_arrays;
    reshaped_arrays.reserve(arrays.size());
    std::transform(arrays.begin(), arrays.end(), std::back_inserter(reshaped_arrays), [axis](const Array& array) {
        return ExpandDims(array, axis);
    });
    return ConcatenateImpl(reshaped_arrays, axis);
}

namespace {

// Defines the backward pass for Split, for both by-sections and by-indices.
void DefineSplitBackward(const Array& ary, const std::vector<Array>& out, int8_t axis_norm) {
    // TODO(hvy): Avoid creating an intermediate vector of reference when BackwardBuilder accepts std::vector<Array>.
    std::vector<ConstArrayRef> out_refs{};
    out_refs.reserve(out.size());
    std::transform(out.begin(), out.end(), std::back_inserter(out_refs), [](const Array& array) { return ConstArrayRef{array}; });

    // TODO(imanishi): Avoid creating shapes of forward outputs;
    std::vector<Shape> shapes;
    shapes.reserve(out.size());
    std::transform(out.begin(), out.end(), std::back_inserter(shapes), [](const Array& array) { return array.shape(); });

    BackwardBuilder bb{"split", ary, out_refs};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([axis_norm, shapes = std::move(shapes), dtype = ary.dtype(), &device = ary.device()](BackwardContext& bctx) {
            std::vector<Array> output_grads;
            output_grads.reserve(bctx.output_count());
            for (size_t i = 0; i < bctx.output_count(); ++i) {
                const absl::optional<Array>& gy = bctx.output_grad(i);
                output_grads.emplace_back(gy.has_value() ? *gy : Zeros(shapes[i], dtype, device));
            }
            bctx.input_grad() = ConcatenateImpl(output_grads, axis_norm);
        });
    }
    bb.Finalize();
}

}  // namespace

std::vector<Array> Split(const Array& ary, int64_t sections, int8_t axis) {
    if (sections < 1) {
        throw DimensionError("Number of sections must be larger than 0.");
    }

    const Shape& in_shape = ary.shape();
    int8_t axis_norm = internal::NormalizeAxis(axis, ary.ndim());
    int64_t in_dim = in_shape[axis_norm];

    if (in_dim % sections != 0) {
        throw DimensionError("Array split does not result in an equal division.");
    }

    Shape out_shape = in_shape;
    int64_t out_dim = in_dim / sections;
    out_shape[axis_norm] = out_dim;
    int64_t out_stride = ary.strides()[axis_norm];
    int64_t out_offset = ary.offset();
    bool is_empty = ary.GetTotalSize() == 0;

    std::vector<Array> out{};
    out.reserve(sections);

    for (int64_t i = 0; i < sections; ++i) {
        out.emplace_back(internal::MakeArray(out_shape, ary.strides(), ary.dtype(), ary.device(), ary.data(), out_offset));

        // Empty arrays should all have offsets of 0 to e.g. avoid out-of-memory errors.
        if (!is_empty) {
            out_offset += out_stride * out_dim;
        }
    }

    DefineSplitBackward(ary, out, axis_norm);

    return out;
}

std::vector<Array> Split(const Array& ary, std::vector<int64_t> indices, int8_t axis) {
    const Shape& in_shape = ary.shape();
    int8_t axis_norm = internal::NormalizeAxis(axis, ary.ndim());
    int64_t in_dim = in_shape[axis_norm];

    // Wrap negative indices.
    std::transform(
            indices.begin(), indices.end(), indices.begin(), [in_dim](int64_t index) { return index >= 0 ? index : index + in_dim; });
    indices.emplace_back(in_dim);

    Shape out_shape = in_shape;
    int64_t out_stride = ary.strides()[axis_norm];
    int64_t out_offset = ary.offset();
    int64_t slice_start = 0;
    bool is_empty = ary.GetTotalSize() == 0;

    std::vector<Array> out{};
    out.reserve(indices.size());

    for (int64_t index : indices) {
        int64_t slice_stop = std::min(in_dim, std::max(int64_t{0}, index));
        int64_t slice_step = slice_stop - slice_start;

        // Update the dimension of interest in the output shape.
        out_shape[axis_norm] = std::max(int64_t{0}, slice_step);

        out.emplace_back(internal::MakeArray(out_shape, ary.strides(), ary.dtype(), ary.device(), ary.data(), out_offset));

        // Empty arrays should all have offsets of 0 to e.g. avoid out-of-memory errors.
        if (!is_empty) {
            out_offset += out_stride * slice_step;
        }

        slice_start = slice_stop;
    }

    DefineSplitBackward(ary, out, axis_norm);

    return out;
}

std::vector<Array> DSplit(const Array& ary, int64_t sections) {
    if (sections < 1) {
        throw DimensionError("Number of sections must be larger than 0.");
    }

    if (ary.ndim() < 3) {
        throw DimensionError("dsplit only works on arrays of 3 or more dimensions.");
    }

    return Split(ary, sections, 2);
}

std::vector<Array> DSplit(const Array& ary, std::vector<int64_t> indices) {
    if (ary.ndim() < 3) {
        throw DimensionError("dsplit only works on arrays of 3 or more dimensions.");
    }

    return Split(ary, std::move(indices), 2);
}

std::vector<Array> VSplit(const Array& ary, int64_t sections) {
    if (sections < 1) {
        throw DimensionError("Number of sections must be larger than 0.");
    }

    if (ary.ndim() < 2) {
        throw DimensionError("vsplit only works on arrays of 2 or more dimensions.");
    }

    return Split(ary, sections, 0);
}

std::vector<Array> VSplit(const Array& ary, std::vector<int64_t> indices) {
    if (ary.ndim() < 2) {
        throw DimensionError("vsplit only works on arrays of 2 or more dimensions.");
    }

    return Split(ary, std::move(indices), 0);
}

std::vector<Array> HSplit(const Array& ary, int64_t sections) {
    if (sections < 1) {
        throw DimensionError("Number of sections must be larger than 0.");
    }

    if (ary.ndim() == 0) {
        throw DimensionError("hsplit only works on arrays of 1 or more dimensions.");
    }

    if (ary.ndim() > 1) {
        return Split(ary, sections, 1);
    }

    return Split(ary, sections, 0);
}

std::vector<Array> HSplit(const Array& ary, std::vector<int64_t> indices) {
    if (ary.ndim() == 0) {
        throw DimensionError("hsplit only works on arrays of 1 or more dimensions.");
    }

    if (ary.ndim() > 1) {
        return Split(ary, std::move(indices), 1);
    }

    return Split(ary, std::move(indices), 0);
}

Array Swapaxes(const Array& a, int8_t axis1, int8_t axis2) {
    Shape shape = a.shape();
    Strides strides = a.strides();

    axis1 = internal::NormalizeAxis(axis1, a.ndim());
    axis2 = internal::NormalizeAxis(axis2, a.ndim());

    std::iter_swap(shape.begin() + axis1, shape.begin() + axis2);
    std::iter_swap(strides.begin() + axis1, strides.begin() + axis2);
    Array out = internal::MakeArray(shape, strides, a.dtype(), a.device(), a.data(), a.offset());

    BackwardBuilder bb{"swapaxes", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([axis1, axis2](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = Swapaxes(gout, axis1, axis2);
        });
    }
    bb.Finalize();

    return out;
}

Array Ravel(const Array& a) { return a.Reshape({a.GetTotalSize()}); }

Array Repeat(const Array& a, int64_t repeats, absl::optional<int8_t> axis) {
    if (repeats < 0) {
        throw DimensionError("repeats must be larger than 0.");
    }

    int8_t target_axis = 0;
    Array target_array;

    if (axis.has_value()) {
        target_axis = internal::NormalizeAxis(*axis, a.ndim());
        target_array = a;
    } else {
        target_array = Reshape(a, Shape({a.shape().GetTotalSize()}));
    }

    Shape broadcast_shape = target_array.shape();
    broadcast_shape.insert(broadcast_shape.begin() + target_axis + 1, repeats);

    Shape reshape_shape = target_array.shape();
    reshape_shape[target_axis] *= repeats;

    Array expanded_array = ExpandDims(target_array, target_axis + 1);
    Array broadcasted_array = BroadcastTo(expanded_array, broadcast_shape);
    Array reshaped_array = Reshape(broadcasted_array, reshape_shape);
    return AsContiguousArray(reshaped_array);
}

Array Repeat(const Array& a, const std::vector<int64_t>& repeats, absl::optional<int8_t> axis) {
    if (repeats.size() == 1) {
        return Repeat(a, repeats[0], axis);
    }

    if (axis.has_value()) {
        int8_t target_axis = internal::NormalizeAxis(*axis, a.ndim());

        if (repeats.size() != static_cast<size_t>(a.shape()[target_axis])) {
            throw DimensionError("The number of repeats must be same with a shape in the axis direction.");
        }

        if (std::any_of(repeats.begin(), repeats.end(), [](int64_t x) -> bool { return x < 0; })) {
            throw DimensionError("repeats must be larger than 0.");
        }

        // TODO(durswd) : should be optimized
        std::vector<Array> output_elements;
        std::vector<Array> splitted = Split(a, a.shape()[target_axis], target_axis);

        for (size_t i = 0; i < splitted.size(); ++i) {
            for (int32_t j = 0; j < repeats[i]; ++j) {
                output_elements.push_back(splitted[i]);
            }
        }

        Array out = Concatenate(output_elements, target_axis);

        return AsContiguousArray(out);
    }

    if (repeats.size() != static_cast<size_t>(a.shape().GetTotalSize())) {
        throw DimensionError("The number of repeats must be same with a shape.");
    }

    Array reshaped = Reshape(a, Shape({a.shape().GetTotalSize()}));
    return Repeat(reshaped, repeats, 0);
}

Array ExpandDims(const Array& a, int8_t axis) {
    Shape shape = a.shape();

    axis = internal::NormalizeAxis(axis, a.ndim() + 1);

    shape.insert(shape.begin() + axis, 1);

    Array out = a.Reshape(shape);

    // A trivial reshape of adding a new axis should just return a view of the input.
    CHAINERX_ASSERT(out.raw_data() == a.raw_data());

    return out;
}

Array Flip(const Array& m, const OptionalAxes& axes) {
    Axes real_axes;
    if (axes.has_value()) {
        real_axes = internal::GetNormalizedAxes(*axes, m.ndim());
    } else {
        for (int8_t i = 0; i < m.ndim(); ++i) {
            real_axes.emplace_back(m.ndim() - i - 1);
        }
    }

    Strides strides = m.strides();
    Shape shape = m.shape();
    int64_t offset = m.offset();
    for (auto axis : real_axes) {
        // last element of that dimension.
        offset += std::max<int64_t>(shape[axis] - 1, 0) * strides[axis];
        if (shape[axis] != 0) {
            strides[axis] = -strides[axis];
        }
    }

    auto is_zero = std::find(shape.begin(), shape.end(), 0);
    if (is_zero != shape.end()) {
        offset = 0;
    }

    Array out = internal::MakeArray(m.shape(), strides, m.dtype(), m.device(), m.data(), offset);

    BackwardBuilder bb{"flip", m, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([real_axes](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = Flip(gout, real_axes);
        });
    }
    bb.Finalize();

    return out;
}

Array Fliplr(const Array& m) {
    if (m.ndim() < 2) {
        throw DimensionError{"Input must be >= 2-d."};
    }
    return Flip(m, Axes{1});
}

Array Flipud(const Array& m) {
    if (m.ndim() < 1) {
        throw DimensionError{"Input must be >= 1-d."};
    }
    return Flip(m, Axes{0});
}

Array AtLeast2D(const Array& x) {
    Array out;

    {
        NoBackpropModeScope scope;

        switch (x.ndim()) {
            case 0:
                out = x.Reshape({1, 1});
                break;
            case 1: {
                Shape shape = x.shape();
                Strides strides = x.strides();
                shape.insert(shape.begin(), 1);
                strides.insert(strides.begin(), 0);
                out = internal::MakeArray(shape, strides, x.dtype(), x.device(), x.data());
            } break;
            default:
                out = x.MakeView();
                break;
        }
    }

    BackwardBuilder bb{"atleast_2d", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([in_shape = x.shape(), ndim = x.ndim()](BackwardContext& bctx) {
            if (ndim <= 1) {
                bctx.input_grad() = bctx.output_grad()->Reshape(in_shape);
            } else {
                bctx.input_grad() = *bctx.output_grad();
            }
        });
    }
    bb.Finalize();

    return out;
}

Array AtLeast3D(const Array& x) {
    Array out;

    {
        NoBackpropModeScope scope;

        switch (x.ndim()) {
            case 0:
                out = x.Reshape({1, 1, 1});
                break;
            case 1: {
                Shape shape = x.shape();
                Strides strides = x.strides();
                shape.insert(shape.begin(), 1);
                shape.insert(shape.end(), 1);
                strides.insert(strides.begin(), 0);
                strides.insert(strides.end(), 0);
                out = internal::MakeArray(shape, strides, x.dtype(), x.device(), x.data());
            } break;
            case 2: {
                Shape shape = x.shape();
                Strides strides = x.strides();
                shape.insert(shape.end(), 1);
                strides.insert(strides.end(), 0);
                out = internal::MakeArray(shape, strides, x.dtype(), x.device(), x.data());
            } break;
            default:
                out = x.MakeView();
                break;
        }
    }

    BackwardBuilder bb{"atleast_3d", x, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([in_shape = x.shape(), ndim = x.ndim()](BackwardContext& bctx) {
            if (ndim <= 2) {
                bctx.input_grad() = bctx.output_grad()->Reshape(in_shape);
            } else {
                bctx.input_grad() = *bctx.output_grad();
            }
        });
    }
    bb.Finalize();

    return out;
}

Array HStack(const std::vector<Array>& arrays) {
    if (arrays.empty()) {
        throw DimensionError{"Need at least one array to stack"};
    }

    if (arrays.front().ndim() <= 1) {
        return Concatenate(arrays, 0);
    }
    return Concatenate(arrays, 1);
}

Array VStack(const std::vector<Array>& arrays) {
    if (arrays.empty()) {
        throw DimensionError{"Need at least one array to stack"};
    }

    std::vector<Array> reshaped_arrays(arrays.size());
    std::transform(arrays.begin(), arrays.end(), reshaped_arrays.begin(), AtLeast2D);

    return Concatenate(reshaped_arrays, 0);
}

Array DStack(const std::vector<Array>& arrays) {
    if (arrays.empty()) {
        throw DimensionError{"Need at least one array to stack"};
    }

    std::vector<Array> reshaped_arrays(arrays.size());
    std::transform(arrays.begin(), arrays.end(), reshaped_arrays.begin(), AtLeast3D);
    return Concatenate(reshaped_arrays, 2);
}

Array Moveaxis(const Array& a, const Axes& source, const Axes& destination) {
    if (source.size() != destination.size()) {
        throw DimensionError{"Invalid Source or Destination Axes"};
    }

    if (source.empty()) {
        return a;
    }

    const Axes& normalized_source = internal::GetNormalizedAxes(source, a.ndim());
    const Axes& normalized_destination = internal::GetNormalizedAxes(destination, a.ndim());

    Axes order, source_axes, destination_axes;
    order.resize(a.ndim());
    source_axes.resize(a.ndim());
    destination_axes.resize(a.ndim());

    std::iota(source_axes.begin(), source_axes.end(), 0);
    std::iota(destination_axes.begin(), destination_axes.end(), 0);

    for (int8_t i = 0; i < source.ndim(); ++i) {
        order[normalized_destination[i]] = normalized_source[i];
        source_axes[normalized_source[i]] = -1;
        destination_axes[normalized_destination[i]] = -1;
    }

    auto source_iter = std::remove(source_axes.begin(), source_axes.end(), -1);
    auto destination_iter = std::remove(destination_axes.begin(), destination_axes.end(), -1);

    int8_t rest_dim = a.ndim() - source.ndim();
    CHAINERX_ASSERT(a.ndim() - destination.ndim() == rest_dim);
    CHAINERX_ASSERT(static_cast<int8_t>(source_iter - source_axes.begin()) == rest_dim);
    CHAINERX_ASSERT(static_cast<int8_t>(destination_iter - destination_axes.begin()) == rest_dim);

    for (int8_t i = 0; i < rest_dim; ++i) {
        order[destination_axes[i]] = source_axes[i];
    }

    return a.Transpose(order);
}

void CopyTo(const Array& dst, const Array& src, CastingMode casting, const Array& where) {
    internal::CheckNoUnsafeInplace(dst, {dst, src, where});

    switch (casting) {
        case CastingMode::kNo:
            if (dst.dtype() != src.dtype()) {
                throw DtypeError{"Source and destination must have same dtype."};
            }
            break;
        default:
            CHAINERX_NEVER_REACH();
    }

    const Array& src_b = src.shape() != dst.shape() ? src.BroadcastTo(dst.shape()) : src;
    const Array& where_b = where.shape() != dst.shape() ? where.BroadcastTo(dst.shape()) : where;

    {
        NoBackpropModeScope scope;
        dst.device().backend().CallKernel<WhereKernel>(where_b, src_b, dst, dst);
    }
}

}  // namespace chainerx
