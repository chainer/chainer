#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/axes.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/elementwise.h"
#include "xchainer/native/reduce.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/reduction_kernel_arg.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/slice.h"

namespace xchainer {
namespace native {

std::shared_ptr<void> NativeDevice::Allocate(size_t bytesize) { return std::make_unique<uint8_t[]>(bytesize); }

void NativeDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    (void)src_device;  // unused
    assert(nullptr != dynamic_cast<NativeDevice*>(&src_device) && "Native device only supports copy between native devices");
    std::memcpy(dst, src, bytesize);
}

void NativeDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    (void)dst_device;  // unused
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

std::shared_ptr<void> NativeDevice::FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    (void)bytesize;  // unused
    return src_ptr;
}

void NativeDevice::Fill(const Array& out, Scalar value) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T& out) { out = value; }
            T value;
        };
        Elementwise<T>(Impl{static_cast<T>(value)}, out);
    });
}

void NativeDevice::Arange(Scalar start, Scalar step, const Array& out) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t i, T& out) { out = start + step * i; }
            T start;
            T step;
        };
        Elementwise<T>(Impl{static_cast<T>(start), static_cast<T>(step)}, out);
    });
}

void NativeDevice::ArgMax(const Array& a, const Axes& axis, const Array& out) {
    assert(std::all_of(axis.begin(), axis.end(), [&a](int8_t i) { return a.shape()[i] > 0; }));
    assert(internal::IsValidReductionShape(a.shape(), axis, out.shape(), false));
    CheckDevicesCompatible(a, out);

    VisitDtype(a.dtype(), [&a, &axis, &out](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            struct MaxAndArgMax {
                T max;
                int64_t argmax;
            };

            MaxAndArgMax Identity() { return {T{}, -1}; }
            MaxAndArgMax MapIn(T in, int64_t index) { return {in, index}; }
            void Reduce(MaxAndArgMax next, MaxAndArgMax& accum) {
                if (accum.argmax < 0 || accum.max < next.max) {
                    accum = next;
                }
            }
            int64_t MapOut(MaxAndArgMax accum) { return accum.argmax; }
        };
        Reduce(MakeReductionKernelArg<T, int64_t>(a, axis, out), Impl{});
    });
}

void NativeDevice::Sum(const Array& a, const Axes& axis, const Array& out) {
    assert(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);

    VisitDtype(out.dtype(), [&a, &axis, &out](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            T Identity() { return T{0}; }
            T MapIn(T in, int64_t /*index*/) { return in; }
            void Reduce(T next, T& accum) { accum += next; }
            T MapOut(T accum) { return accum; }
        };
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), Impl{});
    });
}

void NativeDevice::AMax(const Array& a, const Axes& axis, const Array& out) {
    assert(internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);

    VisitDtype(a.dtype(), [&a, &axis, &out](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            T Identity() { return NumericLimits<T>::LowestOrInf(); }
            T MapIn(T in, int64_t /*index*/) { return in; }
            void Reduce(T next, T& accum) {
                if (std::isnan(next) || accum < next) {
                    accum = next;
                }
            }
            T MapOut(T accum) { return accum; }
        };
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), Impl{});
    });
}

void NativeDevice::Copy(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T a, T& out) { out = a; }
        };
        Elementwise<const T, T>(Impl{}, a, out);
    });
}

void NativeDevice::AsType(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    auto do_astype = [&](auto in_pt, auto out_pt) {
        using InT = typename decltype(in_pt)::type;
        using OutT = typename decltype(out_pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, InT a, OutT& out) { out = static_cast<OutT>(a); }
        };
        Elementwise<const InT, OutT>(Impl{}, a, out);
    };
    VisitDtype(out.dtype(), [&](auto out_pt) { VisitDtype(a.dtype(), do_astype, out_pt); });
}

void NativeDevice::Equal(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 == x2; }
        };
        Elementwise<const T, const T, bool>(Impl{}, x1, x2, out);
    });
}

void NativeDevice::Add(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 + x2; }
        };
        Elementwise<const T, const T, T>(Impl{}, x1, x2, out);
    });
}

void NativeDevice::Subtract(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 - x2; }
        };
        Elementwise<const T, const T, T>(Impl{}, x1, x2, out);
    });
}

void NativeDevice::Multiply(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 * x2; }
        };
        Elementwise<const T, const T, T>(Impl{}, x1, x2, out);
    });
}

void NativeDevice::MultiplyAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T& out) { out = x1 * x2; }
            T x2;
        };
        Elementwise<const T, T>(Impl{static_cast<T>(x2)}, x1, out);
    });
}

void NativeDevice::Divide(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T lhs, T rhs, T& out) { out = lhs / rhs; }
        };
        Elementwise<const T, const T, T>(Impl{}, lhs, rhs, out);
    });
}

void NativeDevice::IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x1, T neg, T& out) { out = x1 < x2 ? pos : neg; }
            T x2;
            T pos;
        };
        Elementwise<const T, const T, T>(Impl{static_cast<T>(x2), static_cast<T>(pos)}, x1, neg, out);
    });
}

void NativeDevice::Dot(const Array& a, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, b, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> a_iarray{a};
        IndexableArray<const T> b_iarray{b};
        IndexableArray<T> out_iarray{out};

        // These asserts have to check iarray instead of the original array, otherwise clang-tidy fails bound-checking.
        assert(a_iarray.ndim() == 2);
        assert(b_iarray.ndim() == 2);
        assert(out_iarray.ndim() == 2);

        int64_t m = a.shape()[0];
        int64_t k = a.shape()[1];
        int64_t n = b.shape()[1];
        assert(b.shape()[0] == k);
        assert(out.shape()[0] == m);
        assert(out.shape()[1] == n);

        // TODO(beam2d): Use BLAS.
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                int64_t out_i[] = {i, j};
                T& out_value = out_iarray[out_i];
                out_value = 0;
                for (int64_t l = 0; l < k; ++l) {
                    int64_t a_i[] = {i, l};
                    int64_t b_i[] = {l, j};
                    out_value += a_iarray[a_i] * b_iarray[b_i];
                }
            }
        }
    });
}

void NativeDevice::Exp(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = std::exp(x); }
        };
        Elementwise<const T, T>(Impl{}, x, out);
    });
}

void NativeDevice::Log(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t /*i*/, T x, T& out) { out = std::log(x); }
        };
        Elementwise<const T, T>(Impl{}, x, out);
    });
}

void NativeDevice::Take(const Array& a, const Array& indices, int8_t axis, const Array& out) {
    CheckDevicesCompatible(a, indices, out);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        IndexableArray<const T> a_iarray{a};
        IndexableArray<T> out_iarray{out};
        IndexableArray<const int64_t> indices_iarray{indices};
        Indexer<> a_indexer{a.shape()};
        Indexer<> out_indexer{out.shape()};
        Indexer<> indices_indexer{indices.shape()};

        int64_t axis_dim = a.shape()[axis];

        // left: set of input dimensions lower than the axis
        // right: set of input dimensions higher than the axis
        Shape left_shape{a.shape().begin(), a.shape().begin() + axis};
        Shape right_shape{a.shape().begin() + (axis + 1), a.shape().end()};
        Shape axis_shape{axis_dim};  // always ndim==1
        Indexer<> left_indexer{left_shape};
        Indexer<> right_indexer{right_shape};
        Indexer<> axis_indexer{axis_shape};

        for (auto it = indices_indexer.It(0); it; ++it) {
            int64_t index = indices_iarray[it];
            if (index < 0) {
                index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
            } else {
                index = index % axis_dim;
            }
            assert(0 <= index);
            assert(index < axis_dim);
            auto it_axis = axis_indexer.It(index);

            for (auto it_left = left_indexer.It(0); it_left; ++it_left) {
                for (auto it_right = right_indexer.It(0); it_right; ++it_right) {
                    auto it_out = out_indexer.At(it_left, it, it_right);
                    auto it_a = a_indexer.At(it_left, it_axis, it_right);
                    out_iarray[it_out] = a_iarray[it_a];
                }
            }
        }
    });
}

void NativeDevice::AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, indices, b);
    assert(a.shape() == out.shape());
    VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        IndexableArray<const T> a_iarray{a};
        IndexableArray<const T> b_iarray{b};
        IndexableArray<const int64_t> indices_iarray{indices};
        IndexableArray<T> out_iarray{out};
        Indexer<> b_indexer{b.shape()};
        Indexer<> indices_indexer{indices.shape()};
        Indexer<> out_indexer{out.shape()};  // indexer for both out_iarray and a_array

        int64_t axis_dim = a.shape()[axis];

        // left: set of input dimensions lower than the axis
        // right: set of input dimensions higher than the axis
        Shape left_shape{a.shape().begin(), a.shape().begin() + axis};
        Shape right_shape{a.shape().begin() + (axis + 1), a.shape().end()};
        Shape axis_shape{axis_dim};  // always ndim==1
        Indexer<> left_indexer{left_shape};
        Indexer<> right_indexer{right_shape};
        Indexer<> axis_indexer{axis_shape};

        // Copy
        for (auto it = out_indexer.It(0); it; ++it) {
            out_iarray[it] = a_iarray[it];
        }

        // Add
        for (auto it = indices_indexer.It(0); it; ++it) {
            int64_t index = indices_iarray[it];
            if (index < 0) {
                index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
            } else {
                index = index % axis_dim;
            }
            assert(0 <= index);
            assert(index < axis_dim);
            auto it_axis = axis_indexer.It(index);

            for (auto it_left = left_indexer.It(0); it_left; ++it_left) {
                for (auto it_right = right_indexer.It(0); it_right; ++it_right) {
                    auto it_out = out_indexer.At(it_left, it_axis, it_right);
                    auto it_b = b_indexer.At(it_left, it, it_right);
                    out_iarray[it_out] += b_iarray[it_b];
                }
            }
        }
    });
}

void NativeDevice::Identity(const Array& out) {
    assert(out.ndim() == 2);
    assert(out.shape()[0] == out.shape()[1]);

    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            explicit Impl(int64_t n) : n_plus_one{n + 1} {}
            void operator()(int64_t i, T& out) { out = i % n_plus_one == 0 ? T{1} : T{0}; }
            int64_t n_plus_one;
        };
        Elementwise<T>(Impl{out.shape()[0]}, out);
    });
}

void NativeDevice::Eye(int64_t k, const Array& out) {
    VisitDtype(out.dtype(), [k, &out](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            Impl(int64_t m, int64_t k) : start{k < 0 ? -k * m : k}, stop{m * (m - k)}, step{m + 1} {}
            void operator()(int64_t i, T& out) { out = start <= i && i < stop && (i - start) % step == 0 ? T{1} : T{0}; }
            int64_t start;
            int64_t stop;
            int64_t step;
        };
        Elementwise<T>(Impl{out.shape()[1], k}, out);
    });
}

void NativeDevice::Diagflat(const Array& v, int64_t k, const Array& out) {
    assert(v.ndim() == 1);
    assert(out.ndim() == 2);

    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // Start indices for the 2-D array axes with applied offset k.
        int64_t row_start{0};
        int64_t col_start{0};

        if (k >= 0) {
            col_start += k;
        } else {
            row_start -= k;
        }

        IndexableArray<const T, 1> v_iarray{v};
        IndexableArray<T, 2> out_iarray{out};
        Indexer<1> v_indexer{v.shape()};
        Indexer<1> out_rows_indexer{Shape{out.shape()[0]}};
        Indexer<1> out_cols_indexer{Shape{out.shape()[1]}};
        Indexer<2> out_indexer{out.shape()};

        // Initialize all elements to 0 first instead of conditionally filling in the diagonal.
        for (auto out_it = out_indexer.It(0); out_it; ++out_it) {
            out_iarray[out_it] = T{0};
        }

        for (auto v_it = v_indexer.It(0); v_it; ++v_it) {
            auto out_rows_it = out_rows_indexer.It(row_start + v_it.raw_index());
            auto out_cols_it = out_cols_indexer.It(col_start + v_it.raw_index());
            auto out_it = out_indexer.At(out_rows_it, out_cols_it);
            out_iarray[out_it] = v_iarray[v_it];
        }
    });
}

void NativeDevice::Linspace(double start, double stop, const Array& out) {
    assert(out.ndim() == 1);
    assert(out.shape()[0] > 0);

    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        struct Impl {
            void operator()(int64_t i, T& out) {
                double value = n == 1 ? start : (start * (n - 1 - i) + stop * i) / (n - 1);
                out = static_cast<T>(value);
            }
            int64_t n;
            double start;
            double stop;
        };

        int64_t n = out.shape()[0];
        Elementwise<T>(Impl{n, start, stop}, out);
    });
}

namespace {

int64_t GetConvOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all) {
    if (cover_all) {
        return (in_dim + pad * 2 - kernel_size + stride - 1) / stride + 1;
    }
    return (in_dim + pad * 2 - kernel_size) / stride + 1;
}

Array Im2Col(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    auto ndim = static_cast<int8_t>(kernel_size.size());  // Number of input image dimensions.
    assert(ndim == static_cast<int8_t>(stride.size()));
    assert(ndim == static_cast<int8_t>(pad.size()));
    assert(ndim + 2 == x.ndim());  // Batch and channel dimensions.

    Device& device = x.device();

    // Create a padded copy of the input image.
    // TODO(hvy): Use the Pad function when implemented.
    Shape padded_shape = x.shape();
    std::vector<ArrayIndex> unpadded_slice{ArrayIndex{Slice{}}, ArrayIndex{Slice{}}};  // All batch and channel dimensions.
    for (int64_t i = 0; i < ndim; ++i) {
        padded_shape[i + 2] += pad[i] * 2 + (cover_all ? stride[i] - 1 : 0);  // Pad on both sides.
        unpadded_slice.emplace_back(Slice{pad[i], pad[i] + x.shape()[i]});
    }
    // TODO(hvy): Allow non-zero padding.
    Array padded_x = Zeros(padded_shape, x.dtype(), device);
    device.Copy(x, padded_x.At(unpadded_slice));

    // Create the output array.
    StackVector<int64_t, kMaxNdim> out_dims;  // Number of patches along each axis
    for (int8_t i = 0; i < ndim; ++i) {
        out_dims.emplace_back(GetConvOutDim(x.shape()[i + 2], kernel_size[i], stride[i], pad[i], cover_all));
        assert(out_dims.back() > 0);
    }

    int64_t batch_size = x.shape()[0];
    int64_t channels = x.shape()[1];

    Shape out_shape{batch_size, channels};
    std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(out_shape));
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));
    Array out = Empty(out_shape, x.dtype(), device);

    // Write to the output array.
    VisitDtype(x.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        Indexer<2> batch_channel_indexer{Shape{batch_size, channels}};
        Indexer<> kernel_indexer{Shape{kernel_size.begin(), kernel_size.end()}};
        Indexer<> out_dims_indexer{Shape{out_dims.begin(), out_dims.end()}};
        Indexer<> x_indexer{padded_x.shape()};
        Indexer<> out_indexer{out.shape()};
        IndexableArray<const T> x_iarray{padded_x};
        IndexableArray<T> out_iarray{out};

        // Indices over input image.
        NdimIndex img_index{ndim};

        for (auto it_kernel = kernel_indexer.It(0); it_kernel; ++it_kernel) {
            for (auto it_out_dims = out_dims_indexer.It(0); it_out_dims; ++it_out_dims) {
                for (int i = 0; i < ndim; ++i) {
                    img_index.index()[i] = it_out_dims.index()[i] * stride[i] + it_kernel.index()[i];
                }

                for (auto it_batch_channel = batch_channel_indexer.It(0); it_batch_channel; ++it_batch_channel) {
                    auto it_x = x_indexer.At(it_batch_channel, img_index);
                    auto it_out = out_indexer.At(it_batch_channel, it_kernel, it_out_dims);

                    // Write the output column value.
                    out_iarray[it_out] = x_iarray[it_x];
                }
            }
        }
    });

    return out;
}

// Returns necessary data for TensorDot for one of the input arrays.
// It is called for both inputs to TensorDot.
//
// It returns a tuple of:
// 0. Permuted axes for transpose, moving axes to be reduced to either front or back of array axes.
// 1. Non-reduced shape dimensions to be used in the output shape of TensorDot.
std::tuple<Axes, Shape> GetTensorDotRollAxes(const Shape& shape, const Axes& reduce_axes, bool reduced_axes_first) {
    bool to_reduce[kMaxNdim]{};  // Initialized with false.
    Shape remain_dims;
    Axes roll_axes;
    for (int8_t i = 0; i < reduce_axes.ndim(); ++i) {
        gsl::at(to_reduce, reduce_axes[i]) = true;
    }

    // There are two steps:
    // A. Insert axes to be reduced to roll_axes.
    // B. Insert non-reduced axes to roll_axes.
    // The order of these steps depends on reduced_axes_first.
    for (int step = 0; step < 2; ++step) {
        if ((step == 0) == reduced_axes_first) {
            // Step A.
            for (int8_t i = 0; i < shape.ndim(); ++i) {
                if (gsl::at(to_reduce, i)) {
                    roll_axes.emplace_back(i);
                }
            }
        } else {
            // Step B.
            for (int8_t i = 0; i < shape.ndim(); ++i) {
                if (!gsl::at(to_reduce, i)) {
                    roll_axes.emplace_back(i);
                    remain_dims.emplace_back(shape[i]);
                }
            }
        }
    }

    return std::make_tuple(roll_axes, remain_dims);
}

Array TensorDot(const Array& a, const Array& b, const Axes& a_axis, const Axes& b_axis) {
    assert(a_axis.ndim() == b_axis.ndim());
    assert(a.ndim() >= a_axis.ndim());
    assert(b.ndim() >= b_axis.ndim());
    int8_t axis_ndim = a_axis.ndim();

    // Compute the product of reduced dimensions and check that corresponding dimensions in a_axis and b_axis are of equal length.
    int64_t axis_total_size = 1;
    for (int8_t i = 0; i < axis_ndim; ++i) {
        int64_t a_dim = a.shape()[a_axis[i]];
        assert(a_dim == b.shape()[b_axis[i]]);
        axis_total_size *= a_dim;
    }

    // Compute necessary data for Dot and Reshape.
    auto a_tup = GetTensorDotRollAxes(a.shape(), a_axis, false);
    auto b_tup = GetTensorDotRollAxes(b.shape(), b_axis, true);
    const Axes& a_roll_axes = std::get<0>(a_tup);
    const Axes& b_roll_axes = std::get<0>(b_tup);
    const Shape& a_remain_dims = std::get<1>(a_tup);
    const Shape& b_remain_dims = std::get<1>(b_tup);
    int64_t a_remain_total_size = a_remain_dims.GetTotalSize();
    int64_t b_remain_total_size = b_remain_dims.GetTotalSize();
    Shape a_shape{a_remain_total_size, axis_total_size};
    Shape b_shape{axis_total_size, b_remain_total_size};

    // Compute the dot product between a and b reshaped to 2-dimensions.
    Shape dot_shape{a_remain_total_size, b_remain_total_size};
    Array dot_out = Empty(dot_shape, a.dtype(), a.device());
    a.device().Dot(a.Transpose(a_roll_axes).Reshape(a_shape), b.Transpose(b_roll_axes).Reshape(b_shape), dot_out);

    // Reshape and return the output array.
    Shape out_shape = a_remain_dims;
    std::copy(b_remain_dims.begin(), b_remain_dims.end(), std::back_inserter(out_shape));
    return dot_out.Reshape(out_shape);
}

}  // namespace

Array NativeDevice::Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    int8_t ndim = w.ndim() - 2;

    // Compute the kernel size from the weight array.
    StackVector<int64_t, kMaxNdim> kernel_size;
    std::copy_n(w.shape().begin() + 2, ndim, std::back_inserter(kernel_size));

    // Convert to colum representation of shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
    Array col = Im2Col(x, kernel_size, stride, pad, cover_all);

    // Compute the tensor dot product of col and w, reducing (channel, k_1, k_2, ..., k_n).
    Axes axes;
    axes.resize(ndim + 1);
    std::iota(axes.begin(), axes.end(), 1);
    Array y = TensorDot(col, w, axes, axes);  // (batch_size, out_1, out_2, ..., out_n, out_channel)

    // Add bias, if given.
    if (b.has_value()) {
        y += *b;
    }

    // Move the out channel axis to the second
    Axes roll_axes;
    roll_axes.resize(y.ndim());
    roll_axes[0] = 0;
    roll_axes[1] = ndim + 1;
    std::iota(roll_axes.begin() + 2, roll_axes.end(), 1);
    Array out = y.Transpose(roll_axes);

    return out;
}

namespace {

int64_t GetConvTransposeOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all = false) {
    if (cover_all) {
        return stride * (in_dim - 1) + kernel_size - stride + 1 - 2 * pad;
    }
    return stride * (in_dim - 1) + kernel_size - 2 * pad;
}

Array Col2Im(
        const Array& col,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size) {
    // Cannot use const due to internal compiler error with gcc 5.4.0.
    int8_t batch_size = col.shape()[0];
    int8_t channels = col.shape()[1];
    auto ndim = static_cast<int8_t>(stride.size());

    Shape padded_shape{batch_size, channels};
    for (int8_t i = 0; i < ndim; ++i) {
        padded_shape.emplace_back(out_size[i] + 2 * pad[i] + stride[i] - 1);
    }
    Array padded_out = Zeros(padded_shape, col.dtype(), col.device());

    // Write to the output array
    VisitDtype(col.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        Indexer<2> batch_channel_indexer{Shape{batch_size, channels}};
        Indexer<> kernel_indexer{Shape{col.shape().begin() + 2, col.shape().begin() + 2 + ndim}};
        Indexer<> in_image_dims_indexer{Shape{col.shape().begin() + 2 + ndim, col.shape().end()}};
        Indexer<> col_indexer{col.shape()};
        Indexer<> padded_out_indexer{padded_shape};
        IndexableArray<const T> col_iarray{col};
        IndexableArray<T> padded_out_iarray{padded_out};

        // Indices over the output image.
        NdimIndex out_image_index{ndim};

        for (auto it_kernel = kernel_indexer.It(0); it_kernel; ++it_kernel) {
            for (auto it_in_image_dims = in_image_dims_indexer.It(0); it_in_image_dims; ++it_in_image_dims) {
                for (int8_t i = 0; i < ndim; ++i) {
                    out_image_index.index()[i] = it_in_image_dims.index()[i] * stride[i] + it_kernel.index()[i];
                }

                for (auto it_batch_channel = batch_channel_indexer.It(0); it_batch_channel; ++it_batch_channel) {
                    auto it_col = col_indexer.At(it_batch_channel, it_kernel, it_in_image_dims);
                    auto it_padded_out = padded_out_indexer.At(it_batch_channel, out_image_index);
                    padded_out_iarray[it_padded_out] += col_iarray[it_col];
                }
            }
        }
    });

    std::vector<ArrayIndex> slice{ArrayIndex{Slice{}}, ArrayIndex{Slice{}}};  // All batch and channel dimensions.
    for (int8_t i = 0; i < ndim; ++i) {
        slice.emplace_back(Slice{pad[i], pad[i] + out_size[i]});
    }
    return padded_out.At(slice);
}

}  // namespace

Array NativeDevice::ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& out_size) {
    Array col = TensorDot(w, x, {0}, {1});  // shape: out_channel, k_1, ..., k_n, batch_size, out_1, ..., out_n
    col = RollAxis(col, x.ndim() - 1);  // batch axis is rolled to the top

    StackVector<int64_t, kMaxNdim> out_size_value;
    if (out_size.has_value()) {
        out_size_value = *out_size;
    } else {
        for (size_t i = 0; i < stride.size(); ++i) {
            out_size_value.emplace_back(GetConvTransposeOutDim(x.shape()[i + 2], w.shape()[i + 2], stride[i], pad[i]));
        }
    }

    Array y = Col2Im(col, stride, pad, out_size_value);  // batch_size, out_channels, out_size...

    // Add bias, if given.
    if (b.has_value()) {
        std::vector<ArrayIndex> slice{NewAxis{}, Slice{}};
        for (size_t i = 0; i < out_size_value.size(); ++i) {
            slice.emplace_back(NewAxis{});
        }
        y += b->At(slice);
    }

    return y;
}

void NativeDevice::Synchronize() {}

}  // namespace native
}  // namespace xchainer
