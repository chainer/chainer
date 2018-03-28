#include "xchainer/routines/manipulation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

#include "xchainer/routines/util.h"

namespace xchainer {

Array Transpose(const Array& a) {
    Shape out_shape{a.shape().rbegin(), a.shape().rend()};
    Strides out_strides{a.strides().rbegin(), a.strides().rend()};
    Array out = internal::MakeArray(out_shape, out_strides, a.dtype(), a.device(), a.data(), a.offset());
    internal::SetUpOpNodes("transpose", {a}, out, {[](const Array& gout, const std::vector<GraphId>&) { return gout.Transpose(); }});
    return out;
}

Array Reshape(const Array& a, const Shape& newshape) {
    const Shape& in_shape = a.shape();
    const Strides& in_strides = a.strides();

    // If the shape is unchanged, just return a view.
    if (in_shape == newshape) {
        return a.MakeView();
    }

    // Check for invalid shape.
    int64_t total_size = in_shape.GetTotalSize();
    if (total_size != newshape.GetTotalSize()) {
        throw DimensionError("Cannot reshape array of size " + std::to_string(total_size) + " into shape " + newshape.ToString());
    }

    int64_t element_size = GetElementSize(a.dtype());
    Strides strides;
    if (total_size == 0) {
        // Calculate the strides for 0-sized array.
        std::vector<int64_t> rev_strides_vec;
        rev_strides_vec.push_back(element_size);
        for (int8_t i = newshape.ndim() - 1; i >= 1; --i) {
            rev_strides_vec.push_back(rev_strides_vec.back() * std::max(int64_t{1}, newshape[i]));
        }
        strides = Strides{rev_strides_vec.rbegin(), rev_strides_vec.rend()};
    } else {
        // Calculate the strides for non-0-sized array.

        // reduced_shape and reduced_strides are the shortest shape and strides which can be convertible from input shape and strides
        // without copy.
        std::vector<int64_t> reduced_shape;
        std::vector<int64_t> reduced_strides;
        if (in_shape.ndim() == 0) {
            // Input shape is (). Treat as if it were (1).
            reduced_shape.push_back(int64_t{1});
            reduced_strides.push_back(element_size);
        } else {
            // Add the first pair
            reduced_shape.reserve(in_shape.ndim());
            reduced_strides.reserve(in_shape.ndim());
            reduced_shape.push_back(in_shape[0]);
            reduced_strides.push_back(in_strides[0]);
            // Reduce the remaining
            for (int8_t i = 1; i < in_shape.ndim(); ++i) {
                int64_t dim = in_shape[i];
                int64_t st = in_strides[i];
                assert(dim > 0);
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
        assert(reduced_shape.size() == reduced_strides.size());
        assert(!reduced_shape.empty());

        // Construct the strides for no-copy reshape.
        // If it's not possible, can_reshape_without_copy will be false.
        bool can_reshape_without_copy = true;
        std::vector<int64_t> strides_vec;
        if (newshape.ndim() > 0) {
            int64_t last_stride = reduced_shape[0] * reduced_strides[0];
            size_t i_dim = 0;
            strides_vec.reserve(newshape.ndim());
            for (int64_t dim : newshape) {
                if (dim == 0) {
                    strides_vec.push_back(last_stride);
                    continue;
                }
                if (i_dim >= reduced_shape.size() || reduced_shape[i_dim] % dim != 0) {
                    strides_vec.clear();
                    can_reshape_without_copy = false;
                    break;
                }
                reduced_shape[i_dim] /= dim;
                last_stride = reduced_shape[i_dim] * reduced_strides[i_dim];
                strides_vec.push_back(last_stride);
                if (reduced_strides[i_dim] == 1) {
                    ++i_dim;
                }
            }
        }

        if (!can_reshape_without_copy) {
            // Reshape without copy is not possible.
            // TODO(niboshi): Implement it
            throw NotImplementedError("Reshape that requires a copy is not implemented yet.");
        }
        assert(strides_vec.size() == newshape.size());

        strides = Strides{strides_vec.begin(), strides_vec.end()};
    }

    Array out = internal::MakeArray(newshape, strides, a.dtype(), a.device(), a.data(), a.offset());
    internal::SetUpOpNodes(
            "reshape", {a}, out, {[in_shape](const Array& gout, const std::vector<GraphId>&) { return gout.Reshape(in_shape); }}, {});

    assert(out.shape() == newshape);
    assert(out.strides().size() == newshape.size());
    return out;
}

Array Squeeze(const Array& a, const nonstd::optional<std::vector<int8_t>>& axis) {
    const Shape& in_shape = a.shape();
    const Strides& in_strides = a.strides();

    std::vector<int64_t> out_shape;
    std::vector<int64_t> out_strides;

    if (axis.has_value()) {
        std::vector<int8_t> sorted_axis = internal::GetSortedAxes(*axis, in_shape.ndim());

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
                    throw DimensionError(os.str());
                }
            } else {
                out_shape.push_back(in_shape[i]);
                out_strides.push_back(in_strides[i]);
            }
        }
    } else {  // All axes are candidates for removal if none are given.
        for (int64_t i = 0; i < in_shape.ndim(); ++i) {
            if (in_shape[i] != 1) {
                out_shape.push_back(in_shape[i]);
                out_strides.push_back(in_strides[i]);
            }
        }
    }

    Array out = in_shape.size() == out_shape.size() ? a
                                                    : internal::MakeArray(
                                                              Shape{out_shape.begin(), out_shape.end()},
                                                              Strides{out_strides.begin(), out_strides.end()},
                                                              a.dtype(),
                                                              a.device(),
                                                              a.data(),
                                                              a.offset());
    internal::SetUpOpNodes(
            "squeeze", {a}, out, {[in_shape](const Array& gout, const std::vector<GraphId>&) { return gout.Reshape(in_shape); }});

    return out;
}

Array BroadcastTo(const Array& array, const Shape& shape) {
    const Shape& in_shape = array.shape();
    const Strides& in_strides = array.strides();

    if (in_shape.size() > shape.size()) {
        throw DimensionError("Cannot broadcast to smaller dimensions");
    }

    std::vector<int64_t> rev_strides;
    rev_strides.reserve(shape.size());
    int8_t i_in = in_shape.ndim() - 1;
    for (int8_t i_out = shape.ndim() - 1; i_out >= 0; --i_out) {
        int64_t out_dim = shape[i_out];
        // If this dimension is to be broadcasted, nonbroadcast_stride is unset.
        // Otherwise, it holds the new stride.
        nonstd::optional<int64_t> nonbroadcast_stride{};
        if (i_in >= 0) {
            int64_t in_dim = in_shape[i_in];
            int64_t in_stride = in_strides[i_in];
            --i_in;
            if (in_dim == 1) {
                // do nothing; broadcast
            } else if (in_dim == out_dim) {
                nonbroadcast_stride = in_stride;
            } else {
                throw DimensionError("Invalid broadcast from " + in_shape.ToString() + " to " + shape.ToString());
            }
        } else {
            // do nothing; broadcast
        }

        if (nonbroadcast_stride.has_value()) {
            // non-broadcast dimension
            rev_strides.push_back(*nonbroadcast_stride);
        } else {
            // broadcast dimension
            rev_strides.push_back(int64_t{0});
        }
    }
    assert(rev_strides.size() == shape.size());
    Array out = internal::MakeArray(
            shape, {rev_strides.rbegin(), rev_strides.rend()}, array.dtype(), array.device(), array.data(), array.offset());

    auto backward_function = [in_shape](const Array& gout, const std::vector<GraphId>&) {
        if (gout.shape() == in_shape) {
            return gout;
        }

        int8_t lead = gout.ndim() - in_shape.ndim();
        std::vector<int8_t> lead_axis(lead);
        std::iota(lead_axis.begin(), lead_axis.end(), int8_t{0});

        std::vector<int8_t> axis{lead_axis};
        for (int8_t i = 0; i < in_shape.ndim(); ++i) {
            if (in_shape[i] == 1) {
                axis.emplace_back(i + lead);
            }
        }
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());  // Sum does not accept axis with duplicate elements

        Array gin = gout.Sum(axis, true);
        if (lead > 0) {
            return gin.Squeeze(lead_axis);
        }
        return gin;
    };
    internal::SetUpOpNodes("broadcast_to", {array}, out, {backward_function});

    return out;
}

}  // namespace xchainer
