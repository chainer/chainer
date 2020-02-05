#include "chainerx/routines/creation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "chainerx/array.h"
#include "chainerx/backend.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/constant.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {
namespace internal {

size_t GetRequiredBytes(const Shape& shape, const Strides& strides, size_t item_size) {
    CHAINERX_ASSERT(shape.ndim() == strides.ndim());

    if (shape.GetTotalSize() == 0) {
        return 0;
    }

    // Calculate the distance between the first and the last element, plus single element size.
    size_t n_bytes = item_size;
    for (int8_t i = 0; i < shape.ndim(); ++i) {
        n_bytes += (shape[i] - 1) * std::abs(strides[i]);
    }
    return n_bytes;
}

Array FromHostData(
        const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, const Strides& strides, int64_t offset, Device& device) {
    auto range = GetDataRange(shape, strides, GetItemSize(dtype));
    // TODO(niboshi): Copy only required region. Currently the whole preceding (offset) region is copied.
    std::shared_ptr<void> device_data = device.FromHostMemory(data, offset + std::get<1>(range));
    return internal::MakeArray(shape, strides, dtype, device, std::move(device_data), offset);
}

Array Empty(const Shape& shape, Dtype dtype, const Strides& strides, Device& device) {
    auto bytesize = GetRequiredBytes(shape, strides, GetItemSize(dtype));
    std::shared_ptr<void> data = device.Allocate(bytesize);
    return MakeArray(shape, strides, dtype, device, std::move(data));
}

Array EmptyReduced(const Shape& shape, Dtype dtype, const Axes& axes, bool keepdims, Device& device) {
    Shape out_shape = ReduceShape(shape, axes, keepdims);
    if (!keepdims) {
        return Empty(out_shape, dtype, device);
    }
    // Set reduced strides of the output array to 0
    Strides out_strides{out_shape, dtype};
    for (int8_t axis : axes) {
        out_strides[axis] = 0;
    }
    return Empty(out_shape, dtype, out_strides, device);
}

}  // namespace internal

Array FromContiguousHostData(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device) {
    return internal::FromHostData(shape, dtype, data, {shape, dtype}, 0, device);
}

Array FromData(
        const Shape& shape,
        Dtype dtype,
        const std::shared_ptr<void>& data,
        const absl::optional<Strides>& strides,
        int64_t offset,
        Device& device) {
    return internal::MakeArray(
            shape, strides.value_or(Strides{shape, dtype}), dtype, device, device.MakeDataFromForeignPointer(data), offset);
}

Array Empty(const Shape& shape, Dtype dtype, Device& device) {
    auto bytesize = static_cast<size_t>(shape.GetTotalSize() * GetItemSize(dtype));
    std::shared_ptr<void> data = device.Allocate(bytesize);
    return internal::MakeArray(shape, Strides{shape, dtype}, dtype, device, std::move(data));
}

Array Full(const Shape& shape, Scalar fill_value, Dtype dtype, Device& device) {
    Array array = Empty(shape, dtype, device);
    array.Fill(fill_value);
    return array;
}

Array Full(const Shape& shape, Scalar fill_value, Device& device) {
    return Full(shape, fill_value, internal::GetDefaultDtype(fill_value.kind()), device);
}

Array Zeros(const Shape& shape, Dtype dtype, Device& device) { return Full(shape, 0, dtype, device); }

Array Ones(const Shape& shape, Dtype dtype, Device& device) { return Full(shape, 1, dtype, device); }

Array Arange(Scalar start, Scalar stop, Scalar step, Dtype dtype, Device& device) {
    // TODO(hvy): Simplify comparison if Scalar::operator== supports dtype conversion.
    if (static_cast<double>(step) == 0.0) {
        throw ChainerxError("Cannot create an arange array with 0 step size.");
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
    device.backend().CallKernel<ArangeKernel>(start, step, out);
    return out;
}

Array Arange(Scalar start, Scalar stop, Scalar step, Device& device) {
    // TODO(hvy): Type promote instead of using the dtype of step.
    return Arange(start, stop, step, internal::GetDefaultDtype(step.kind()), device);
}

Array Arange(Scalar start, Scalar stop, Dtype dtype, Device& device) { return Arange(start, stop, 1, dtype, device); }

Array Arange(Scalar start, Scalar stop, Device& device) {
    // TODO(hvy): Type promote dtype instead of using the dtype of stop.
    return Arange(start, stop, 1, internal::GetDefaultDtype(stop.kind()), device);
}

Array Arange(Scalar stop, Dtype dtype, Device& device) { return Arange(0, stop, 1, dtype, device); }

Array Arange(Scalar stop, Device& device) { return Arange(0, stop, 1, internal::GetDefaultDtype(stop.kind()), device); }

Array EmptyLike(const Array& a, Device& device) { return Empty(a.shape(), a.dtype(), device); }

Array FullLike(const Array& a, Scalar fill_value, Device& device) { return Full(a.shape(), fill_value, a.dtype(), device); }

Array ZerosLike(const Array& a, Device& device) { return Zeros(a.shape(), a.dtype(), device); }

Array OnesLike(const Array& a, Device& device) { return Ones(a.shape(), a.dtype(), device); }

Array Copy(const Array& a) {
    Array out = EmptyLike(a, a.device());
    {
        NoBackpropModeScope scope{};
        a.device().backend().CallKernel<CopyKernel>(a, out);
    }

    BackwardBuilder bb{"copy", a, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
    }
    bb.Finalize();

    CHAINERX_ASSERT(out.IsContiguous());
    return out;
}

// Creates the identity array.
Array Identity(int64_t n, Dtype dtype, Device& device) {
    if (n < 0) {
        throw DimensionError{"Negative dimensions are not allowed"};
    }

    Array out = Empty(Shape{n, n}, dtype, device);
    {
        NoBackpropModeScope scope{};
        device.backend().CallKernel<IdentityKernel>(out);
    }
    return out;
}

Array Eye(int64_t n, absl::optional<int64_t> m, absl::optional<int64_t> k, absl::optional<Dtype> dtype, Device& device) {
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
    {
        NoBackpropModeScope scope{};
        device.backend().CallKernel<EyeKernel>(k.value(), out);
    }
    return out;
}

Array AsContiguous(const Array& a, Dtype dtype) {
    if (a.IsContiguous() && a.dtype() == dtype) {
        return a;
    }

    Array out = Empty(a.shape(), dtype, a.device());
    {
        NoBackpropModeScope scope{};
        // Note: In CopyKernel, Input Array Elements are casted to the type of Output Array.
        a.device().backend().CallKernel<CopyKernel>(a.AsGradStopped(), out);
    }

    if (GetKind(dtype) == DtypeKind::kFloat) {
        BackwardBuilder bb{"ascontiguousarray", a, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([src_dtype = a.dtype()](BackwardContext& bctx) {
                const Array& gout = *bctx.output_grad();
                bctx.input_grad() = gout.AsType(src_dtype, false);
            });
        }
        bb.Finalize();
    }

    CHAINERX_ASSERT(out.IsContiguous());
    CHAINERX_ASSERT(out.shape() == a.shape());
    CHAINERX_ASSERT(out.dtype() == dtype);
    return out;
}

Array AsContiguousArray(const Array& a, absl::optional<Dtype> dtype) {
    Dtype src_dt = a.dtype();
    Dtype dt = dtype.value_or(src_dt);

    if (a.IsContiguous() && src_dt == dt) {
        if (a.ndim() == 0) {
            return a.Reshape(Shape{1});
        }
        return a;
    }

    Array out = AsContiguous(a, dt);
    if (a.ndim() == 0) {
        out = out.Reshape({1});
    }
    return out;
}

Array Diag(const Array& v, int64_t k) {
    Array out{};
    Device& device = v.device();

    int8_t ndim = v.ndim();
    if (ndim == 1) {
        // Return a square matrix with filled diagonal.
        int64_t n = v.shape()[0] + std::abs(k);
        out = Empty(Shape{n, n}, v.dtype(), device);
        {
            NoBackpropModeScope scope{};
            device.backend().CallKernel<DiagflatKernel>(v, k, out);
        }
    } else if (ndim == 2) {
        // Return the diagonal as a 1D array.
        int64_t rows = v.shape()[0];
        int64_t cols = v.shape()[1];
        int64_t n = std::min(rows, cols);
        int64_t offset{};
        if (k >= 0) {
            offset = k * v.strides()[1];
            if (cols <= k + n - 1) {
                n = std::max(int64_t{0}, cols - k);
            }
        } else {
            offset = -k * v.strides()[0];
            if (rows <= -k + n - 1) {
                n = std::max(int64_t{0}, rows + k);
            }
        }
        out = internal::MakeArray(Shape{n}, Strides{v.strides()[0] + v.strides()[1]}, v.dtype(), device, v.data(), v.offset() + offset);
    } else {
        throw DimensionError{"Input must be 1D or 2D."};
    }

    BackwardBuilder bb{"diag", v, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([k](BackwardContext& bctx) {
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = Diag(gout, k);
        });
    }
    bb.Finalize();

    return out;
}

Array Diagflat(const Array& v, int64_t k) {
    // TODO(hvy): Use Ravel or Flatten when implemented instead of Reshape.
    return Diag(v.Reshape({v.GetTotalSize()}), k);
}

// Creates a 1-d array with evenly spaced numbers.
Array Linspace(Scalar start, Scalar stop, absl::optional<int64_t> num, bool endpoint, absl::optional<Dtype> dtype, Device& device) {
    static const int64_t kDefaultNum = 50;

    // Always default to float type.
    // Similar behavior to numpy
    // Ref: https://github.com/numpy/numpy/issues/8597
    Dtype dtype_a = dtype.value_or(internal::GetDefaultDtype(chainerx::DtypeKind::kFloat));
    int64_t num_a = num.value_or(kDefaultNum);

    if (num_a < 0) {
        throw ChainerxError{"Number of samples, ", num_a, ", must be non-negative"};
    }

    Array out = Empty(Shape{num_a}, dtype_a, device);
    if (num_a > 0) {
        auto start_value = static_cast<double>(start);
        auto stop_value = static_cast<double>(stop);
        if (!endpoint) {
            stop_value = start_value + (stop_value - start_value) * (num_a - 1) / num_a;
        }
        {
            NoBackpropModeScope scope{};
            device.backend().CallKernel<LinspaceKernel>(start_value, stop_value, out);
        }
    }
    return out;
}

std::vector<Array> Meshgrid(const std::vector<Array>& arrays, MeshgridIndexingMode mode) {
    Shape shape;
    Shape broadcast_shape;
    std::vector<Shape> broadcasted_array_shapes;
    std::vector<Array> grid_arrays;

    // special cases
    // similar behavior to numpy.
    if (arrays.empty()) {
        return grid_arrays;
    }

    if (arrays.size() == 1) {
        grid_arrays.emplace_back(arrays[0].Flatten());
        return grid_arrays;
    }

    grid_arrays.reserve(arrays.size());
    broadcasted_array_shapes.reserve(arrays.size());

    // Algo
    //
    // Step 1: Reshape/View each array as broadcastable based
    // on number of input vectors.
    // Eg. For tuple of vectors (n1, n2, n3)
    // where ni is length of that vector.
    // After this step for Vector 1 , we will reshape it as
    // (n1, 1, 1) , Vector 2 as (1, n2, 1)
    //
    // Step 2: Broadcast each vector to the shape
    // if (indexing == "ij") -> (n1, n2, n3)
    // else if (indexing == "xy") -> (n2, n1, n3)
    // Note : For "xy" only n1 and n2 swap their places
    //        all others are same as "ij"

    // Step 1
    for (const Array& array : arrays) {
        shape.emplace_back(1);
        broadcast_shape.emplace_back(array.GetTotalSize());
    }

    // Shape for each array based on number of arrays.
    for (size_t i = 0; i < arrays.size(); ++i) {
        Shape temp_shape{shape.begin(), shape.end()};
        temp_shape[i] = arrays[i].GetTotalSize();
        broadcasted_array_shapes.emplace_back(temp_shape);
    }

    // Referred from numpy documentation and source.
    if (mode == MeshgridIndexingMode::kCartesian) {
        std::swap(broadcasted_array_shapes[0][0], broadcasted_array_shapes[0][1]);
        std::swap(broadcasted_array_shapes[1][0], broadcasted_array_shapes[1][1]);
        std::swap(broadcast_shape[0], broadcast_shape[1]);
    }

    std::vector<Array> reshaped_arrays;
    reshaped_arrays.reserve(arrays.size());
    for (size_t i = 0; i < arrays.size(); ++i) {
        reshaped_arrays.emplace_back(arrays[i].Reshape(broadcasted_array_shapes[i]));
    }

    // Step 2
    for (const Array& reshaped_array : reshaped_arrays) {
        grid_arrays.emplace_back(reshaped_array.BroadcastTo(broadcast_shape));
    }

    return grid_arrays;
}

Array Tri(int64_t n, absl::optional<int64_t> m, absl::optional<int64_t> k, absl::optional<Dtype> dtype, Device& device) {
    if (!m.has_value()) {
        m = n;
    }
    if (!k.has_value()) {
        k = 0;
    }
    if (!dtype.has_value()) {
        dtype = Dtype::kFloat32;
    }
    // NumPy returns 0-sized array for the input with negative dimensions.
    // This is a flaw in NumPy's implementation. Other array creation routines raise an error for negative dimensions.
    if (n < 0 || m < 0) {
        throw DimensionError{"Negative dimensions are not allowed"};
    }

    Array out = Empty({n, m.value()}, dtype.value(), device);
    {
        NoBackpropModeScope scope{};
        device.backend().CallKernel<TriKernel>(k.value(), out);
    }
    return out;
}

Array Tril(const Array& m, int64_t k = 0) {
    Array out = Empty(m.shape(), m.dtype(), m.device());
    {
        NoBackpropModeScope scope{};
        Array mask{};
        if (m.ndim() >= 2) {
            mask = Tri(m.shape()[m.ndim() - 2], m.shape()[m.ndim() - 1], k, Dtype::kBool, m.device());
        } else {
            mask = Tri(m.shape()[0], m.shape()[0], k, Dtype::kBool, m.device());
        }
        out = Where(mask, m, 0);
    }

    BackwardBuilder bb{"tril", m, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([ndim = m.ndim(), k](BackwardContext& bctx) {
            if (ndim == 1) {
                throw DimensionError{"ChainerX Tril backward is not implemented for 1-dimensional arrays."};
            }
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = Tril(gout, k);
        });
    }
    bb.Finalize();

    return out;
}

Array Triu(const Array& m, int64_t k = 0) {
    Array out = Empty(m.shape(), m.dtype(), m.device());
    {
        NoBackpropModeScope scope{};
        Array mask{};
        if (m.ndim() >= 2) {
            mask = Tri(m.shape()[m.ndim() - 2], m.shape()[m.ndim() - 1], k - 1, Dtype::kBool, m.device());
        } else {
            mask = Tri(m.shape()[0], m.shape()[0], k - 1, Dtype::kBool, m.device());
        }
        out = Where(mask, 0, m);
    }

    BackwardBuilder bb{"triu", m, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([ndim = m.ndim(), k](BackwardContext& bctx) {
            if (ndim == 1) {
                throw DimensionError{"ChainerX Triu backward is not implemented for 1-dimensional arrays."};
            }
            const Array& gout = *bctx.output_grad();
            bctx.input_grad() = Triu(gout, k);
        });
    }
    bb.Finalize();

    return out;
}

}  // namespace chainerx
