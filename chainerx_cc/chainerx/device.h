#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <nonstd/optional.hpp>

#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/constant.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

class Array;
enum class AveragePoolPadMode;

class MaxPoolForwardBackward {
public:
    virtual ~MaxPoolForwardBackward() = default;
    virtual Array Forward(const Array& x) = 0;
    virtual Array Backward(const Array& gout) = 0;
    virtual Array DoubleBackward(const Array& ggx) = 0;
};

class AveragePoolForwardBackward {
public:
    virtual ~AveragePoolForwardBackward() = default;
    virtual Array Forward(const Array& x) = 0;
    virtual Array Backward(const Array& gout) = 0;
};

class BatchNormForwardBackward {
public:
    virtual ~BatchNormForwardBackward() = default;
    virtual Array Forward(const Array& x, const Array& gamma, const Array& beta) = 0;
    virtual std::array<Array, 3> Backward(const Array& gout) = 0;
};

class GenericBatchNormForwardBackward : public BatchNormForwardBackward {
public:
    GenericBatchNormForwardBackward(const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, Axes axis);

    Array Forward(const Array& x, const Array& gamma, const Array& beta) override;
    std::array<Array, 3> Backward(const Array& gout) override;

protected:
    void SetForwardResults(Array x, Array gamma, Array x_mean, Array x_inv_std);

    const Array& running_mean() { return running_mean_; }
    const Array& running_var() { return running_var_; }
    Scalar eps() { return eps_; }
    Scalar decay() { return decay_; }
    const Axes& axis() { return axis_; }

    // Forward results.
    const Array& x() { return *x_; }
    const Array& gamma() { return *gamma_; }
    const Array& x_mean() { return *x_mean_; }
    const Array& x_inv_std() { return *x_inv_std_; }

private:
    const Array& running_mean_;
    const Array& running_var_;
    Scalar eps_;
    Scalar decay_;
    Axes axis_;

    // TODO(niboshi): Fix header dependency order and hold arrays directly.
    std::shared_ptr<Array> x_;
    std::shared_ptr<Array> gamma_;
    std::shared_ptr<Array> x_mean_;
    std::shared_ptr<Array> x_inv_std_;
};

// Device base class.
// Note that these member functions may be called from the framework or user code.
class Device {
public:
    virtual ~Device() = default;

    Device(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(const Device&) = delete;
    Device& operator=(Device&&) = delete;

    // Allocates a memory chunk on this device.
    virtual std::shared_ptr<void> Allocate(size_t bytesize) = 0;

    // Makes an array data pointer from a foreign pointer without copying.
    // May throw an error if the foreign pointer is invalid for this device.
    virtual std::shared_ptr<void> MakeDataFromForeignPointer(const std::shared_ptr<void>& data) { return data; }

    // Copies the data between devices.
    // The other device may or may not be the same as this device.
    // The caller must guarantee that:
    // - Data transfer between the devices is supported by this backend.
    //   That is, Backend::SupportsTransfer must return true for the devices.
    // - Memory regions are not overlapped.
    virtual void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) = 0;
    virtual void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) = 0;

    // Transfers the data from the specified device to this device.
    // It is usually preceded by a call to Backend::SupportsTransfer(), thus this function can assume transfer between the devices are
    // supported.
    //
    // It returns a pointer to the allocated memory.
    virtual std::shared_ptr<void> TransferDataFrom(
            Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) = 0;

    // Transfers the data from this device to the specified device.
    // It is usually preceded by a call to Backend::SupportsTransfer(), thus this function can assume transfer between the devices are
    // supported.
    //
    // It returns a pointer to the allocated memory.
    virtual std::shared_ptr<void> TransferDataTo(
            Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) = 0;

    // Creates a data buffer filled with the specified data on this device.
    //
    // It may allocate a new memory or return an alias.
    // src_ptr must reside in the host memory.
    virtual std::shared_ptr<void> FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) = 0;

    virtual void Fill(const Array& out, Scalar value) = 0;

    virtual void Arange(Scalar start, Scalar step, const Array& out) = 0;

    virtual void ArgMax(const Array& a, const Axes& axis, const Array& out) = 0;

    // Calculate the sum of an array.
    // It will be summed over the specified axes.
    // `axis` must be normalized so that
    // - it has only positive values,
    // - it is sorted, and
    // - it has no duplicated values.
    // Otherwise, the behavior is undefined.
    virtual void Sum(const Array& a, const Axes& axis, const Array& out) = 0;

    // Calculates the maximum along specified axes.
    // See Sum() for the explanation of arguments.
    virtual void AMax(const Array& src, const Axes& axis, const Array& out) = 0;

    // Copies the elements from one array to the other.
    //
    // The arrays must match in shape and dtype and need to reside on this device.
    virtual void Copy(const Array& a, const Array& out) = 0;

    // Casts the elements from one array to the other dtype, and store into the other.
    virtual void AsType(const Array& a, const Array& out) = 0;

    virtual void Equal(const Array& x1, const Array& x2, const Array& out) = 0;
    virtual void NotEqual(const Array& x1, const Array& x2, const Array& out) = 0;
    virtual void Greater(const Array& x1, const Array& x2, const Array& out) = 0;
    virtual void GreaterEqual(const Array& x1, const Array& x2, const Array& out) = 0;
    virtual void LogicalNot(const Array& x1, const Array& out) = 0;

    virtual void Add(const Array& x1, const Array& x2, const Array& out) = 0;
    virtual void AddAS(const Array& x1, Scalar x2, const Array& out) = 0;

    virtual void Subtract(const Array& x1, const Array& x2, const Array& out) = 0;
    virtual void SubtractAS(const Array& x1, Scalar x2, const Array& out) = 0;

    virtual void Multiply(const Array& x1, const Array& x2, const Array& out) = 0;
    virtual void MultiplyAS(const Array& x1, Scalar x2, const Array& out) = 0;

    virtual void FloorDivide(const Array& x1, const Array& x2, const Array& out) = 0;
    virtual void FloorDivideAS(const Array& x1, Scalar x2, const Array& out) = 0;

    virtual void Divide(const Array& x1, const Array& x2, const Array& out) = 0;
    virtual void DivideAS(const Array& x1, Scalar x2, const Array& out) = 0;

    // Compares x1 and x2 and assign either pos or neg according to the result.
    //
    // Formally, it calculates: out = x1 < x2 ? pos : neg
    virtual void IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) = 0;

    // Compares x1 and x2 and assign either pos or neg according to the result.
    //
    // Formally, it calculates: out = x1 > x2 ? pos : neg
    virtual void IfGreaterElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) = 0;

    virtual void Tanh(const Array& x, const Array& out) = 0;

    // Matrix multiplication. All the operands are matrices (i.e., two-dimensional arrays).
    // Let the shapes of `a` and `b` be `(M, K)` and `(L, N)`, respectively.
    // Then, it must hold that `K == L` and the shape of `out` must be `(M, N)`.
    // Otherwise, the behavior is undefined.
    virtual void Dot(const Array& a, const Array& b, const Array& out) = 0;

    virtual void Exp(const Array& x, const Array& out) = 0;
    virtual void Log(const Array& x, const Array& out) = 0;

    virtual void Sqrt(const Array& x, const Array& out) = 0;

    virtual void IsNan(const Array& x, const Array& out) = 0;
    virtual void IsInf(const Array& x, const Array& out) = 0;

    // Takes elements specified by indices from an array.
    // Indices that are out of bounds are wrapped around.
    //
    // `axis` must be within [0, a.ndim()).
    // `indices` must have dtype kind of either kInt or kUInt.
    virtual void Take(const Array& a, const Array& indices, int8_t axis, const Array& out) = 0;

    // Adds each slice of `b` along the axis `axis` to `a`'s corresponding slices, specified by `indices`.
    // The result is assigned in `out. Input arrays `a`, `indices`, and `b` are not altered.
    //
    // TODO(niboshi): This function may be replaced with full-featured assignable advanced indexing.
    //
    // `axis` must be within [0, b.ndim()).
    // `indices` must have dtype kind of either kInt or kUInt.
    virtual void AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) = 0;

    // Creates the identity array.
    // out must be a square 2-dim array.
    virtual void Identity(const Array& out) = 0;

    // Creates a 2-dimensional array with ones along the k-th diagonal and zeros elsewhere.
    // out must be a square 2-dim array.
    virtual void Eye(int64_t k, const Array& out) = 0;

    virtual void Diagflat(const Array& v, int64_t k, const Array& out) = 0;

    // Creates an evenly spaced 1-d array.
    // `out.ndim()` must be 1 with at least 1 elements.
    virtual void Linspace(double start, double stop, const Array& out) = 0;

    // Computes the n-dimensional convolution.
    //
    // x: (batch_size, in_channels, in_1, in_2, ..., in_n)
    // w: (out_channels, in_channels, k_1, k_2, ..., k_n)
    // b: (out_channels)
    //
    // Returns an array of shape (batch_size, out_channels, out_1, out_2, ..., out_n).
    virtual Array Conv(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) = 0;

    virtual Array ConvGradWeight(
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) = 0;

    // Computes the n-dimensional transposed convolution.
    //
    // x: (batch_size, in_channels, in_1, in_2, ..., in_n)
    // w: (in_channels, out_channels, k_1, k_2, ..., k_n)
    // b: (out_channels)
    //
    // Returns an array of shape (batch_size, out_channels, out_1, out_2, ..., out_n).
    virtual Array ConvTranspose(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& out_size) = 0;

    virtual std::unique_ptr<MaxPoolForwardBackward> GetMaxPoolForwardBackward(
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) = 0;

    virtual std::unique_ptr<AveragePoolForwardBackward> GetAveragePoolForwardBackward(
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            AveragePoolPadMode pad_mode) = 0;

    virtual std::unique_ptr<BatchNormForwardBackward> GetBatchNormForwardBackward(
            const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, const Axes& axis) {
        return std::make_unique<GenericBatchNormForwardBackward>(running_mean, running_var, eps, decay, axis);
    }

    virtual Array FixedBatchNorm(
            const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis);

    virtual void Synchronize() = 0;

    // TODO(sonots): optimize string concat
    std::string name() const { return backend_.GetName() + ":" + std::to_string(index_); }

    Backend& backend() const { return backend_; }
    Context& context() const { return backend_.context(); }
    int index() const { return index_; }

    // Throws an exception if array devices are incompatible, else does nothing.
    template <typename... Arrays>
    void CheckDevicesCompatible(const Array& first, const Arrays&... rest) {
        CheckDevicesCompatible(first);
        CheckDevicesCompatible(rest...);
    }

    void CheckDevicesCompatible(const Array& array);

protected:
    Device(Backend& backend, int index) : backend_{backend}, index_{index} {}

private:
    Backend& backend_;
    int index_;
};

namespace internal {

Device* GetDefaultDeviceNoExcept() noexcept;

}  // namespace internal

// Gets the default device. If the default device is null in this thread, it sets and returns the "native:0" device of the default context.
Device& GetDefaultDevice();

// Sets thread local device.
//
// Raises ContextError if context mismatches between given device and default context.
void SetDefaultDevice(Device* device);

void CheckEqual(const Device& lhs, const Device& rhs);

// Scope object that switches the default device by RAII.
class DeviceScope {
public:
    DeviceScope() : orig_{internal::GetDefaultDeviceNoExcept()}, exited_{false} {}
    explicit DeviceScope(Device& device) : DeviceScope{} { SetDefaultDevice(&device); }

    // TODO(hvy): Maybe unnecessary.
    explicit DeviceScope(Backend* backend, int index = 0) : DeviceScope{backend->GetDevice(index)} {}

    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
    DeviceScope& operator=(DeviceScope&&) = delete;

    DeviceScope(DeviceScope&& other) : orig_(other.orig_), exited_(other.exited_) { other.exited_ = true; }

    ~DeviceScope() { Exit(); }

    // Explicitly recovers the original device. It will invalidate the scope object so that dtor will do nothing.
    void Exit() {
        if (!exited_) {
            SetDefaultDevice(orig_);
            exited_ = true;
        }
    }

private:
    Device* orig_;
    bool exited_;
};

}  // namespace chainerx
