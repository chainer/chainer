#include "xchainer/routines/creation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace internal {

size_t GetRequiredBytes(const Shape& shape, const Strides& strides, size_t element_size) {
    assert(shape.ndim() == strides.ndim());

    if (shape.GetTotalSize() == 0) {
        return 0;
    }

    // Calculate the distance between the first and the last element, plus single element size.
    size_t total_bytes = element_size;
    for (int8_t i = 0; i < shape.ndim(); ++i) {
        total_bytes += (shape[i] - 1) * std::abs(strides[i]);
    }
    return total_bytes;
}

Array FromHostData(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, const Strides& strides, Device& device) {
    auto bytesize = GetRequiredBytes(shape, strides, GetElementSize(dtype));
    std::shared_ptr<void> device_data = device.FromHostMemory(data, bytesize);
    return MakeArray(shape, strides, dtype, device, device_data);
}

Array FromContiguousHostData(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device) {
    return FromHostData(shape, dtype, data, {shape, dtype}, device);
}

Array Empty(const Shape& shape, Dtype dtype, const Strides& strides, Device& device) {
    auto bytesize = GetRequiredBytes(shape, strides, GetElementSize(dtype));
    std::shared_ptr<void> data = device.Allocate(bytesize);
    return MakeArray(shape, strides, dtype, device, data);
}

}  // namespace internal

Array Empty(const Shape& shape, Dtype dtype, Device& device) {
    auto bytesize = static_cast<size_t>(shape.GetTotalSize() * GetElementSize(dtype));
    std::shared_ptr<void> data = device.Allocate(bytesize);
    return internal::MakeArray(shape, Strides{shape, dtype}, dtype, device, data);
}

Array Full(const Shape& shape, Scalar fill_value, Dtype dtype, Device& device) {
    Array array = Empty(shape, dtype, device);
    array.Fill(fill_value);
    return array;
}

Array Full(const Shape& shape, Scalar fill_value, Device& device) { return Full(shape, fill_value, fill_value.dtype(), device); }

Array Zeros(const Shape& shape, Dtype dtype, Device& device) { return Full(shape, 0, dtype, device); }

Array Ones(const Shape& shape, Dtype dtype, Device& device) { return Full(shape, 1, dtype, device); }

Array Arange(Scalar start, Scalar stop, Scalar step, Dtype dtype, Device& device) {
    // TODO(hvy): Simplify comparison if Scalar::operator== supports dtype conversion.
    if (step == Scalar{0, step.dtype()}) {
        throw XchainerError("Cannot create an arange array with 0 step size.");
    }

    // Compute the size of the output.
    auto start_value = static_cast<double>(start);
    auto stop_value = static_cast<double>(stop);
    auto step_value = static_cast<double>(step);
    if (step_value < 0) {
        std::swap(start_value, stop_value);
        step_value *= -1;
    }
    auto size = std::max(int64_t{0}, static_cast<int64_t>(std::ceil((stop_value - start_value) / step_value)));
    if (size > 2 && dtype == Dtype::kBool) {
        throw DtypeError{"Cannot create an arange array of booleans with size larger than 2."};
    }

    Array out = Empty({size}, dtype, device);
    device.Arange(start, step, out);
    return out;
}

Array Arange(Scalar start, Scalar stop, Scalar step, Device& device) {
    // TODO(hvy): Type promote instead of using the dtype of step.
    return Arange(start, stop, step, step.dtype(), device);
}

Array Arange(Scalar start, Scalar stop, Dtype dtype, Device& device) { return Arange(start, stop, 1, dtype, device); }

Array Arange(Scalar start, Scalar stop, Device& device) {
    // TODO(hvy): Type promote dtype instead of using the dtype of stop.
    return Arange(start, stop, 1, stop.dtype(), device);
}

Array Arange(Scalar stop, Dtype dtype, Device& device) { return Arange(0, stop, 1, dtype, device); }

Array Arange(Scalar stop, Device& device) { return Arange(0, stop, 1, stop.dtype(), device); }

Array EmptyLike(const Array& a, Device& device) { return Empty(a.shape(), a.dtype(), device); }

Array FullLike(const Array& a, Scalar fill_value, Device& device) { return Full(a.shape(), fill_value, a.dtype(), device); }

Array ZerosLike(const Array& a, Device& device) { return Zeros(a.shape(), a.dtype(), device); }

Array OnesLike(const Array& a, Device& device) { return Ones(a.shape(), a.dtype(), device); }

Array Copy(const Array& a) {
    // No graph will be disconnected.
    Array out = a.AsConstant({}, CopyKind::kCopy);
    assert(out.IsContiguous());
    return out;
}

// Creates the identity array.
Array Identity(int64_t n, Dtype dtype, Device& device) {
    if (n < 0) {
        throw DimensionError{"Negative dimensions are not allowed"};
    }

    Array out = Empty(Shape{n, n}, dtype, device);
    device.Identity(out);
    return out;
}

Array Eye(int64_t n, nonstd::optional<int64_t> m, nonstd::optional<int64_t> k, nonstd::optional<Dtype> dtype, Device& device) {
    if (!m.has_value()) {
        m = n;
    }
    if (!k.has_value()) {
        k = 0;
    }
    if (!dtype.has_value()) {
        dtype = Dtype::kFloat64;
    }
    if (n < 0 || m < 0) {
        throw DimensionError{"Negative dimensions are not allowed"};
    }

    Array out = Empty({n, m.value()}, dtype.value(), device);
    device.Eye(k.value(), out);
    return out;
}

Array AsContiguousArray(const Array& a, const nonstd::optional<Dtype>& dtype) {
    Dtype src_dt = a.dtype();
    Dtype dt = dtype.value_or(src_dt);

    if (a.IsContiguous() && src_dt == dt) {
        return a;
    }

    Array out = Empty(a.shape(), dt, a.device());
    a.device().AsType(a, out);

    if (GetKind(dt) == DtypeKind::kFloat && GetKind(src_dt) == DtypeKind::kFloat) {
        internal::SetUpOpNodes("ascontiguousarray", {a}, out, {[src_dt](const Array& gout, const std::vector<GraphId>&) {
                                   return gout.AsType(src_dt, false);
                               }});
    }
    assert(out.IsContiguous());
    return out;
}

Array Diag(const Array& v, int64_t k, Device& device) {
    int8_t ndim = v.ndim();
    if (ndim != 1 && ndim != 2) {
        throw DimensionError{"Input must be 1D or 2D."};
    }

    Shape out_shape{};

    if (ndim == 1) {
        // Return a square matrix with filled diagonal.
        int64_t n = v.GetTotalSize() + std::abs(k);
        out_shape.emplace_back(n);
        out_shape.emplace_back(n);
    } else if (ndim == 2) {
        // Return a 1D array, an extracted diagonal.
        int64_t rows = v.shape()[0];
        int64_t cols = v.shape()[1];
        int64_t n = std::min(rows, cols);
        if (k >= 0 && cols <= k + n - 1) {
            n = cols - k;
        } else if (k < 0 && rows >= k - n + 1) {
            n = rows + k;
        }
        out_shape.emplace_back(std::max(int64_t{0}, n));
    }

    Array out = Empty(out_shape, v.dtype(), device);
    device.Diag(v, k, out);
    return out;
}

Array Diagflat(const Array& v, int64_t k, Device& device) {
    // TODO(hvy): Use Ravel or Flatten when implemented instead of Reshape.
    return Diag(v.Reshape({v.GetTotalSize()}), k, device);
}

// Creates a 1-d array with evenly spaced numbers.
Array Linspace(
        Scalar start,
        Scalar stop,
        const nonstd::optional<int64_t>& num,
        bool endpoint,
        const nonstd::optional<Dtype>& dtype,
        Device& device) {
    static const int64_t kDefaultNum = 50;

    // TODO(niboshi): Determine dtype_a from both dtypes of start and stop.
    Dtype dtype_a = dtype.value_or(start.dtype());
    int64_t num_a = num.value_or(kDefaultNum);

    if (num_a < 0) {
        throw XchainerError{"Number of samples, ", num_a, ", must be non-negative"};
    }

    Array out = Empty(Shape{num_a}, dtype_a, device);
    if (num_a > 0) {
        auto start_value = static_cast<double>(start);
        auto stop_value = static_cast<double>(stop);
        if (!endpoint) {
            stop_value = start_value + (stop_value - start_value) * (num_a - 1) / num_a;
        }
        device.Linspace(start_value, stop_value, out);
    }
    return out;
}

}  // namespace xchainer
