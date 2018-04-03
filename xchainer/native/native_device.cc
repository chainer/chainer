#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace native {
namespace internal {

template <typename T>
std::tuple<IndexableArray<const T>, IndexableArray<T>, Indexer<>, Indexer<>, Indexer<>> PrepareIndexableArraysForReduction(
        const Array& src, const std::vector<int8_t>& axis, const Array& out) {
    // True if some axes are reduced but kept in output as 1-dim axes.
    // Corresponding to keepdim argument in Array::Sum().
    bool has_kept_dims = out.ndim() + static_cast<int64_t>(axis.size()) != src.ndim();

    // Prepare axis mappings
    std::vector<int64_t> reduce_shape;  // Reduction dimensions
    std::vector<int8_t> out_axis_map;   // Mapping from effective output indices to actual output indices
    std::vector<int64_t> new_out_shape;
    // (Here "effective output indices" means source indices minus reduction indices.)

    // Example (in the case of has_kept_dims=false):
    // - src.shape():      (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out.shape():      (12, 14, 16)
    // - reduce_shape:     (13, 15)
    // - out_axis_map:     (0, 1, 2)
    // - new_out_shape:    (12, 14, 16)
    // Example (in the case of has_kept_dims=true):
    // - src.shape():      (12, 13, 14, 15, 16)
    // - axis:             (1, 3)
    // - out.shape():      (12, 1, 14, 1, 16)
    // - reduce_shape:     (13, 15)
    // - out_axis_map:     (0, 2, 4)
    // - new_out_shape:    (12, 14, 16)

    reduce_shape.reserve(axis.size());
    out_axis_map.reserve(out.shape().size());
    new_out_shape.reserve(out.shape().size());
    {
        size_t i_axis = 0;
        size_t i_out_axis = 0;
        for (int8_t i = 0; i < src.shape().ndim(); ++i) {
            if (i_axis < axis.size() && i == axis[i_axis]) {
                // i is to be reduced
                reduce_shape.push_back(src.shape()[i]);
                ++i_axis;
                if (has_kept_dims) {
                    ++i_out_axis;
                }
            } else {
                // i is not to be reduced
                int64_t out_dim = out.shape()[i_out_axis];
                if (out_dim != 1) {
                    out_axis_map.push_back(static_cast<int8_t>(i_out_axis));
                    new_out_shape.push_back(out_dim);
                }
                ++i_out_axis;
            }
        }
        assert(i_out_axis == out.shape().size());
        assert(i_axis == axis.size());
    }
    assert(reduce_shape.size() == axis.size());
    assert(out_axis_map.size() <= src.shape().size() - axis.size());  // Inequality because 1-dim axes are eliminated.
    assert(out_axis_map.size() == new_out_shape.size());

    // Calculate source axis permutation
    std::vector<int8_t> axis_permutes;
    axis_permutes.reserve(src.shape().size());
    {
        size_t i_reduce = 0;
        for (int8_t i = 0; i < src.ndim(); ++i) {
            if (i_reduce < axis.size() && i == axis[i_reduce]) {
                ++i_reduce;
            } else {
                if (src.shape()[i] != 1) {
                    axis_permutes.push_back(i);
                }
            }
        }
    }
    std::copy(axis.begin(), axis.end(), std::back_inserter(axis_permutes));
    assert(axis_permutes.size() <= src.shape().size());  // Inequality because 1-dim axes are eliminated.

    // Calculate new source shape
    std::vector<int64_t> new_src_shape;
    new_src_shape.reserve(axis_permutes.size());
    for (int8_t i : axis_permutes) {
        new_src_shape.push_back(src.shape()[i]);
    }

    // 1-dim axes must be eliminated
    assert(std::find(new_src_shape.begin(), new_src_shape.end(), 1) == new_src_shape.end());
    assert(std::find(new_out_shape.begin(), new_out_shape.end(), 1) == new_out_shape.end());

    // Check postconditions and return
    auto tup = std::make_tuple(
            IndexableArray<const T>{src}.Permute(axis_permutes),           // src indexable array
            IndexableArray<T>{out}.Permute(out_axis_map),                  // out indexable array
            Indexer<>{Shape{new_src_shape.begin(), new_src_shape.end()}},  // src indexer
            Indexer<>{Shape{new_out_shape.begin(), new_out_shape.end()}},  // out indexer
            Indexer<>{Shape{reduce_shape.begin(), reduce_shape.end()}});   // reduce indexer
    assert(std::get<0>(tup).ndim() == std::get<2>(tup).ndim());
    assert(std::get<1>(tup).ndim() == std::get<3>(tup).ndim());
    assert(std::get<0>(tup).ndim() == std::get<3>(tup).ndim() + std::get<4>(tup).ndim());
    return tup;
}
}  // namespace internal

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

void NativeDevice::Fill(const Array& out, Scalar value) {
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

void NativeDevice::Sum(const Array& src, const std::vector<int8_t>& axis, const Array& out) {
    // keepdims denotes the corresponding argument in Array::Sum().
    assert(out.ndim() == src.ndim() - static_cast<int64_t>(axis.size()) ||  // keepdims=false
           out.ndim() == src.ndim());                                       // keepdims=true
    CheckDevicesCompatible(src, out);

    VisitDtype(src.dtype(), [&src, &axis, &out](auto pt) {
        using T = typename decltype(pt)::type;

        // Prepare indexable arrays and indexers
        auto tup = internal::PrepareIndexableArraysForReduction<T>(src, axis, out);
        IndexableArray<const T>& src_iarray = std::get<0>(tup);
        IndexableArray<T>& out_iarray = std::get<1>(tup);
        Indexer<>& src_indexer = std::get<2>(tup);
        Indexer<>& out_indexer = std::get<3>(tup);
        Indexer<>& reduce_indexer = std::get<4>(tup);

        // Iterate over output dimensions
        for (int64_t i_out = 0; i_out < out_indexer.total_size(); ++i_out) {
            out_indexer.Set(i_out);
            T sum_value = 0;
            gsl::span<int64_t> src_index = gsl::make_span(src_indexer.index(), src.shape().size());

            // Set output indices in the corresponding indices (out_axis) in src_index.
            for (int8_t i_out_dim = 0; i_out_dim < out_indexer.ndim(); ++i_out_dim) {
                src_index[i_out_dim] = out_indexer.index()[i_out_dim];
            }

            // Iterate over reduction dimensions, reducing into a single output value.
            for (int64_t i_reduce = 0; i_reduce < reduce_indexer.total_size(); ++i_reduce) {
                reduce_indexer.Set(i_reduce);
                // Set reduction indices in the corresponding indices (axis) in src_index.
                for (int8_t i_reduce_dim = 0; i_reduce_dim < reduce_indexer.ndim(); ++i_reduce_dim) {
                    src_index[out_indexer.ndim() + i_reduce_dim] = reduce_indexer.index()[i_reduce_dim];
                }

                sum_value += src_iarray[src_indexer];
            }
            out_iarray[out_indexer] = sum_value;
        }
    });
}

void NativeDevice::Copy(const Array& src, const Array& out) {
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

void NativeDevice::Equal(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<bool> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] == rhs_iarray[indexer];
        }
    });
}

void NativeDevice::Add(const Array& lhs, const Array& rhs, const Array& out) {
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

void NativeDevice::Subtract(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] - rhs_iarray[indexer];
        }
    });
}

void NativeDevice::Mul(const Array& lhs, Scalar rhs, const Array& out) {
    CheckDevicesCompatible(lhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] * static_cast<T>(rhs);
        }
    });
}

void NativeDevice::Mul(const Array& lhs, const Array& rhs, const Array& out) {
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

void NativeDevice::IfLessElse(const Array& lhs, Scalar rhs, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(lhs, neg, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> neg_iarray{neg};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};
        T rhs_value{rhs};
        T pos_value{pos};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] < rhs_value ? pos_value : neg_iarray[indexer];
        }
    });
}

void NativeDevice::Dot(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};

        // These asserts have to check iarray instead of the original array, otherwise clang-tidy fails bound-checking.
        assert(lhs_iarray.ndim() == 2);
        assert(rhs_iarray.ndim() == 2);
        assert(out_iarray.ndim() == 2);

        int64_t m = lhs.shape()[0];
        int64_t k = lhs.shape()[1];
        int64_t n = rhs.shape()[1];
        assert(rhs.shape()[0] == k);
        assert(out.shape()[0] == m);
        assert(out.shape()[1] == n);

        // TODO(beam2d): Use BLAS.
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                int64_t out_i[] = {i, j};
                T& out_value = out_iarray[out_i];
                out_value = 0;
                for (int64_t l = 0; l < k; ++l) {
                    int64_t lhs_i[] = {i, l};
                    int64_t rhs_i[] = {l, j};
                    out_value += lhs_iarray[lhs_i] * rhs_iarray[rhs_i];
                }
            }
        }
    });
}

void NativeDevice::Synchronize() {}

}  // namespace native
}  // namespace xchainer
