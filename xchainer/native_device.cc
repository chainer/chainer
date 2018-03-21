#include "xchainer/native_device.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/scalar.h"

namespace xchainer {

std::shared_ptr<void> NativeDevice::Allocate(size_t bytesize) { return std::make_unique<uint8_t[]>(bytesize); }

void NativeDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    assert(nullptr != dynamic_cast<NativeDevice*>(&src_device) && "Native device only supports copy between native devices");
    std::memcpy(dst, src, bytesize);
}

void NativeDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    assert(nullptr != dynamic_cast<NativeDevice*>(&dst_device) && "Native device only supports copy between native devices");
    std::memcpy(dst, src, bytesize);
}

std::shared_ptr<void> NativeDevice::TransferDataFrom(
        Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFrom(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, src_device);
    return dst_ptr;
}

std::shared_ptr<void> NativeDevice::TransferDataTo(
        Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    return dst_device.TransferDataFrom(*this, src_ptr, offset, bytesize);
}

std::shared_ptr<void> NativeDevice::FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    (void)bytesize;  // unused
    return src_ptr;
}

void NativeDevice::Fill(Array& out, Scalar value) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        T c_value{value};

        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{out.shape()};
        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = c_value;
        }
    });
}

void NativeDevice::Sum(const Array& src, const std::vector<int8_t>& axis, Array& out) {
    // keepdims denotes the corresponding argument in Array::Sum().
    assert(out.ndim() == src.ndim() - static_cast<int64_t>(axis.size()) ||  // keepdims=false
           out.ndim() == src.ndim());                                       // keepdims=true
    CheckDevicesCompatible(src, out);

    VisitDtype(src.dtype(), [&src, &axis, &out](auto pt) {
        using T = typename decltype(pt)::type;
        const Shape& out_shape = out.shape();

        // True if some axes are reduced but kept in output as 1-dim axes.
        // Corresponding to keepdim argument in Array::Sum().
        bool has_kept_dims = out.ndim() + static_cast<int64_t>(axis.size()) != src.ndim();

        // In the following logic, output dimensions are first iterated over with `out_indexer`,
        // and then reduction dimensions with `reduce_indexer` in nested manner.
        // `src_indexer` is composed from `out_indexer` and `reduce_indexer` to point a single source value.

        // Prepare axis mappings
        std::vector<int64_t> reduce_shape_vec;  // Reduction dimensions
        std::vector<int8_t> src_axis_map;       // Mapping from effective output indices to src indices
        std::vector<int8_t> out_axis_map;       // Mapping from effective output indices to actual output indices
        // (Here "effective output indices" means source indices minus reduction indices.)

        // Example (in the case of has_kept_dims=false):
        // - src.shape():      (12, 13, 14, 15, 16)
        // - axis:             (1, 3)
        // - out.shape():      (12, 14, 16)
        // - reduce_shape_vec: (13, 15)
        // - src_axis_map:     (0, 2, 4)
        // - out_axis_map:     (0, 1, 2)
        // Example (in the case of has_kept_dims=true):
        // - src.shape():      (12, 13, 14, 15, 16)
        // - axis:             (1, 3)
        // - out.shape():      (12, 1, 14, 1, 16)
        // - reduce_shape_vec: (13, 15)
        // - src_axis_map:     (0, 2, 4)
        // - out_axis_map:     (0, 2, 4)

        reduce_shape_vec.reserve(axis.size());
        src_axis_map.reserve(out.shape().size());
        out_axis_map.reserve(out.shape().size());
        {
            size_t i_axis = 0;
            size_t i_out_axis = 0;
            for (int8_t i = 0; i < src.shape().ndim(); ++i) {
                if (i_axis < axis.size() && i == axis[i_axis]) {
                    // i is to be reduced
                    reduce_shape_vec.push_back(src.shape()[i]);
                    ++i_axis;
                    if (has_kept_dims) {
                        ++i_out_axis;
                    }
                } else {
                    // i is not to be reduced
                    src_axis_map.push_back(i);
                    out_axis_map.push_back(static_cast<int8_t>(i_out_axis));
                    ++i_out_axis;
                }
            }
            assert(i_out_axis == out.shape().size());
            assert(i_axis == axis.size());
        }
        assert(reduce_shape_vec.size() == axis.size());
        assert(src_axis_map.size() == src.shape().size() - axis.size());
        assert(out_axis_map.size() == src.shape().size() - axis.size());

        // Calculate sum
        IndexableArray<const T> src_iarray{src};
        IndexableArray<T> out_iarray{out};
        Indexer<> src_indexer{src.shape()};
        Indexer<> reduce_indexer{Shape{reduce_shape_vec.begin(), reduce_shape_vec.end()}};
        Indexer<> out_indexer{out_shape};

        // Iterate over output dimensions
        for (int64_t i_out = 0; i_out < out_indexer.total_size(); ++i_out) {
            out_indexer.Set(i_out);
            T sum_value = 0;
            gsl::span<int64_t> src_index = gsl::make_span(src_indexer.index(), src.shape().size());

            // Set output indices in the corresponding indices (out_axis) in src_index.
            for (size_t i_out_dim = 0; i_out_dim < src_axis_map.size(); ++i_out_dim) {
                src_index[src_axis_map[i_out_dim]] = out_indexer.index()[out_axis_map[i_out_dim]];
            }

            // Iterate over reduction dimensions, reducing into a single output value.
            for (int64_t i_reduce = 0; i_reduce < reduce_indexer.total_size(); ++i_reduce) {
                reduce_indexer.Set(i_reduce);
                // Set reduction indices in the corresponding indices (axis) in src_index.
                for (int8_t i_reduce_dim = 0; i_reduce_dim < static_cast<int8_t>(axis.size()); ++i_reduce_dim) {
                    src_index[axis[i_reduce_dim]] = reduce_indexer.index()[i_reduce_dim];
                }

                sum_value += src_iarray[src_indexer];
            }
            out_iarray[out_indexer] = sum_value;
        }
    });
}

void NativeDevice::Copy(const Array& src, Array& out) {
    CheckDevicesCompatible(src, out);
    VisitDtype(src.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> src_iarray{src};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{src.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = src_iarray[indexer];
        }
    });
}

void NativeDevice::Add(const Array& lhs, const Array& rhs, Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] + rhs_iarray[indexer];
        }
    });
}

void NativeDevice::Mul(const Array& lhs, const Array& rhs, Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] * rhs_iarray[indexer];
        }
    });
}

void NativeDevice::Synchronize() {}

}  // namespace xchainer
