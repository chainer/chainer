#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <absl/types/optional.h>
#include <absl/types/span.h>

#include "chainerx/array_body.h"
#include "chainerx/array_fwd.h"
#include "chainerx/array_index.h"
#include "chainerx/array_node.h"
#include "chainerx/array_repr.h"
#include "chainerx/axes.h"
#include "chainerx/constant.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/enum.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/scalar.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {
namespace internal {

BackpropId GetArrayBackpropId(const Array& array, const absl::optional<BackpropId>& backprop_id);

Array MakeArray(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset = 0);

inline const std::shared_ptr<ArrayBody>& GetArrayBody(const Array& array);

inline std::shared_ptr<ArrayBody>&& MoveArrayBody(Array&& array);

}  // namespace internal

// The user interface of multi-dimensional arrays.
//
// This wraps an ArrayBody, providing accessors, an interface for graph operations and differentiable operations.
class Array {
public:
    Array() = default;

    ~Array() = default;

    // TODO(hvy): Consider making this contructor private and prohibit body from being null (assert that given body is not null).
    explicit Array(std::shared_ptr<internal::ArrayBody> body) : body_{std::move(body)} {
        if (body_ == nullptr) {
            throw ChainerxError{"Cannot create an array from null."};
        }
    }

    // Copy constructor that copies the pointer to the body instead of the body itself.
    //
    // Use MakeView if you want to clone the body.
    Array(const Array& other) = default;
    Array(Array&& other) = default;

    // Assign operators just replace the body. They do not copy data between arrays.
    Array& operator=(const Array&) = default;
    Array& operator=(Array&& other) = default;

    Array operator-() const;

    Array operator==(const Array& rhs) const;
    Array operator!=(const Array& rhs) const;
    Array operator>(const Array& rhs) const;
    Array operator>=(const Array& rhs) const;
    Array operator<(const Array& rhs) const;
    Array operator<=(const Array& rhs) const;

    Array& operator+=(const Array& rhs);
    Array& operator+=(Scalar rhs);
    Array& operator-=(const Array& rhs);
    Array& operator-=(Scalar rhs);
    Array& operator*=(const Array& rhs);
    Array& operator*=(Scalar rhs);
    Array& operator/=(const Array& rhs);
    Array& operator/=(Scalar rhs);
    Array& operator%=(const Array& rhs);
    Array& operator%=(Scalar rhs);
    Array& operator&=(const Array& rhs);
    Array& operator&=(Scalar rhs);
    Array& operator|=(const Array& rhs);
    Array& operator|=(Scalar rhs);
    Array& operator^=(const Array& rhs);
    Array& operator^=(Scalar rhs);
    Array& operator<<=(const Array& rhs);
    Array& operator<<=(Scalar rhs);
    Array& operator>>=(const Array& rhs);
    Array& operator>>=(Scalar rhs);

    const Array& operator+=(const Array& rhs) const;
    const Array& operator+=(Scalar rhs) const;
    const Array& operator-=(const Array& rhs) const;
    const Array& operator-=(Scalar rhs) const;
    const Array& operator*=(const Array& rhs) const;
    const Array& operator*=(Scalar rhs) const;
    const Array& operator/=(const Array& rhs) const;
    const Array& operator/=(Scalar rhs) const;
    const Array& operator%=(const Array& rhs) const;
    const Array& operator%=(Scalar rhs) const;
    const Array& operator&=(const Array& rhs) const;
    const Array& operator&=(Scalar rhs) const;
    const Array& operator|=(const Array& rhs) const;
    const Array& operator|=(Scalar rhs) const;
    const Array& operator^=(const Array& rhs) const;
    const Array& operator^=(Scalar rhs) const;
    const Array& operator<<=(const Array& rhs) const;
    const Array& operator<<=(Scalar rhs) const;
    const Array& operator>>=(const Array& rhs) const;
    const Array& operator>>=(Scalar rhs) const;

    Array operator+(const Array& rhs) const;
    Array operator+(Scalar rhs) const;
    Array operator-(const Array& rhs) const;
    Array operator-(Scalar rhs) const;
    Array operator*(const Array& rhs) const;
    Array operator*(Scalar rhs) const;
    Array operator/(const Array& rhs) const;
    Array operator/(Scalar rhs) const;
    Array operator%(const Array& rhs) const;
    Array operator%(Scalar rhs) const;
    Array operator&(const Array& rhs) const;
    Array operator&(Scalar rhs) const;
    Array operator|(const Array& rhs) const;
    Array operator|(Scalar rhs) const;
    Array operator^(const Array& rhs) const;
    Array operator^(Scalar rhs) const;
    Array operator<<(const Array& rhs) const;
    Array operator<<(Scalar rhs) const;
    Array operator>>(const Array& rhs) const;
    Array operator>>(Scalar rhs) const;

    // Returns a view selected with the indices.
    Array At(const std::vector<ArrayIndex>& indices) const;

    // Returns a transposed view of the array.
    Array Transpose(const OptionalAxes& axes = absl::nullopt) const;

    // Returns a reshaped array.
    // TODO(niboshi): Support shape with dimension -1.
    Array Reshape(const Shape& newshape) const;

    // Returns a squeezed array with unit-length axes removed.
    //
    // If no axes are specified, all axes of unit-lengths are removed.
    // If no axes can be removed, an array with aliased data is returned.
    Array Squeeze(const OptionalAxes& axis = absl::nullopt) const;

    // Interchange two axes of an array.
    Array Swapaxes(int8_t axis1, int8_t axis2) const;

    // Broadcasts the array to the specified shape.
    // Returned array is always a view to this array.
    Array BroadcastTo(const Shape& shape) const;

    // Returns the indices of the maximum values along the given axis.
    Array ArgMax(const OptionalAxes& axis = absl::nullopt) const;

    // Returns the indices of the minimum values along the given axis.
    Array ArgMin(const OptionalAxes& axis = absl::nullopt) const;

    // Returns a sum of the array.
    // If `axis` is set, it will be summed over the specified axes.
    // Otherwise, it will be summed over all the existing axes.
    // Note: When implementing chainerx::Sum(), be careful of the semantics of the default value of `keepdims`. See NumPy documentation.
    Array Sum(const OptionalAxes& axis = absl::nullopt, bool keepdims = false) const;

    // Returns the maximum value of the array.
    // If `axis` is set, the maximum value is chosen along the specified axes.
    // Otherwise, all the elements are searched at once.
    Array Max(const OptionalAxes& axis = absl::nullopt, bool keepdims = false) const;

    // Returns the minimum value of the array.
    // If `axis` is set, the minimum value is chosen along the specified axes.
    // Otherwise, all the elements are searched at once.
    Array Min(const OptionalAxes& axis = absl::nullopt, bool keepdims = false) const;

    Array Mean(const OptionalAxes& axis = absl::nullopt, bool keepdims = false) const;

    Array Var(const OptionalAxes& axis = absl::nullopt, bool keepdims = false) const;

    Array All(const OptionalAxes& axis = absl::nullopt, bool keepdims = false) const;

    Array Any(const OptionalAxes& axis = absl::nullopt, bool keepdims = false) const;

    // Returns a dot product of the array with another one.
    Array Dot(const Array& b) const;

    // Takes elements specified by indices from the array.
    //
    // TODO(niboshi): Support Scalar and StackVector as indices.
    // TODO(niboshi): Support axis=None behavior in NumPy.
    // TODO(niboshi): Support indices dtype other than int64.
    Array Take(const Array& indices, int8_t axis) const;

    // Creates a copy.
    // It will be connected to all the graphs.
    // It will be always C-contiguous.
    Array Copy() const;

    // Creates a view.
    // It creates a new array node and connects graphs.
    Array MakeView() const;

    // Transfers the array to another device. It will be connected to all the graphs.
    //
    // If the destination is the same device, an array with aliased data is returned.
    // Otherwise, a C-contiguous Array will be created on the target device.
    // TODO(niboshi): Currently control over whether to make an alias is not supported.
    Array ToDevice(Device& dst_device) const;

    // Transfer the array to the native device. It will be connected to all the graphs.
    //
    // This is a wrapper function which calls Array::ToDevice with the native:0 device.
    // See also: Array::ToDevice();
    Array ToNative() const;

    // Creates a copy or a view. It will be disconnected from all the graphs.
    // If `kind` is `CopyKind::kCopy`, the returned array will be always C-contiguous.
    Array AsGradStopped(CopyKind kind = CopyKind::kView) const;

    // Creates a copy or a view. It will be disconnected from the specified graphs.
    // If `kind` is `CopyKind::kCopy`, the returned array will be always C-contiguous.
    Array AsGradStopped(absl::Span<const BackpropId> backprop_ids, CopyKind kind = CopyKind::kView) const;
    Array AsGradStopped(std::initializer_list<const BackpropId> backprop_ids, CopyKind kind = CopyKind::kView) const {
        return AsGradStopped(absl::MakeConstSpan(backprop_ids.begin(), backprop_ids.end()), kind);
    }

    // Casts to a specified type.
    // By default, always returns a newly allocated array. If `copy` is false,
    // and the dtype requirement is satisfied, the input array is returned instead of a copy.
    Array AsType(Dtype dtype, bool copy = true) const;

    void Fill(Scalar value) const;

    // Returns the gradient of the array.
    //
    // ChainerxError is thrown if the array is constant with respect to the computation for the specified backprop ID.
    // ChainerxError is thrown if the array is not flagged as requiring gradient.
    // This function ignores no/force-backprop mode.
    const absl::optional<Array>& GetGrad(const absl::optional<BackpropId>& backprop_id = absl::nullopt) const;

    // Sets the gradient of the array.
    // This function also flags the array as requiring gradient, so that preceding GetGrad() can return the gradient.
    //
    // ChainerxError is thrown if the array is constant with respect to the computation for the specified backprop ID.
    // This function ignores no/force-backprop mode.
    void SetGrad(Array grad, const absl::optional<BackpropId>& backprop_id = absl::nullopt) const;

    // Clears the gradient of the array if set.
    // This function does not change the state of the array other than that. For example, if the array is flagged as requiring gradient,
    // that will not change.
    //
    // ChainerxError is thrown if the array is constant with respect to the computation for the specified backprop ID.
    // This function ignores no/force-backprop mode.
    void ClearGrad(const absl::optional<BackpropId>& backprop_id = absl::nullopt) const;

    // Returns whether the array needs to backprop.
    //
    // If no-backprop mode is set with respect to the specified backprop ID, this function returns false.
    bool IsBackpropRequired(const absl::optional<BackpropId>& backprop_id = absl::nullopt) const;
    bool IsBackpropRequired(AnyGraph any_graph) const;

    // Returns whether the array is flagged to compute the gradient during backprop.
    //
    // This function ignores no/force-backprop mode.
    bool IsGradRequired(const absl::optional<BackpropId>& backprop_id = absl::nullopt) const;

    // Flags the array to compute the gradient during backprop.
    // If the array is constant with respect to the computation of the backprop ID, this function makes the array non-constant.
    //
    // This function ignores no/force-backprop mode.
    const Array& RequireGrad(const absl::optional<BackpropId>& backprop_id = absl::nullopt) const {
        return RequireGradImpl(*this, backprop_id);
    }

    Array& RequireGrad(const absl::optional<BackpropId>& backprop_id = absl::nullopt) { return RequireGradImpl(*this, backprop_id); }

    int64_t GetTotalSize() const { return body_->GetTotalSize(); }

    int64_t GetNBytes() const { return body_->GetNBytes(); }

    int64_t GetItemSize() const { return body_->GetItemSize(); }

    bool IsContiguous() const { return body_->IsContiguous(); }

    std::string ToString() const;

    Context& context() const { return body_->device().context(); }

    Dtype dtype() const { return body_->dtype(); }

    Device& device() const { return body_->device(); }

    int8_t ndim() const { return body_->ndim(); }

    const Shape& shape() const { return body_->shape(); }

    const Strides& strides() const { return body_->strides(); }

    const std::shared_ptr<void>& data() const { return body_->data(); }

    void* raw_data() const { return body_->data().get(); }

    int64_t offset() const { return body_->offset(); }

private:
    friend Array internal::MakeArray(
            const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);
    friend const std::shared_ptr<internal::ArrayBody>& internal::GetArrayBody(const Array& array);
    friend std::shared_ptr<internal::ArrayBody>&& internal::MoveArrayBody(Array&& array);

    Array(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset = 0);

    template <typename T>
    static T& RequireGradImpl(T& array, const absl::optional<BackpropId>& backprop_id);

    std::shared_ptr<internal::ArrayBody> body_;
};

Array operator+(Scalar lhs, const Array& rhs);
Array operator-(Scalar lhs, const Array& rhs);
Array operator*(Scalar lhs, const Array& rhs);
Array operator/(Scalar lhs, const Array& rhs);
Array operator%(Scalar lhs, const Array& rhs);

Array operator<<(Scalar lhs, const Array& rhs);
Array operator>>(Scalar lhs, const Array& rhs);

namespace internal {

inline const std::shared_ptr<ArrayBody>& GetArrayBody(const Array& array) { return array.body_; }

inline std::shared_ptr<ArrayBody>&& MoveArrayBody(Array&& array) { return std::move(array.body_); }

std::vector<std::shared_ptr<ArrayBody>> MoveArrayBodies(std::vector<Array>&& arrays);

std::vector<std::shared_ptr<ArrayBody>> MoveArrayBodies(std::vector<absl::optional<Array>>&& arrays);

}  // namespace internal

void DebugDumpComputationalGraph(
        std::ostream& os,
        const Array& array,
        const absl::optional<BackpropId>& backprop_id,
        int indent = 0,
        const std::vector<std::pair<ConstArrayRef, std::string>>& array_name_map = {});

}  // namespace chainerx
