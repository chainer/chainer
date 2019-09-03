#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/graph.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"

namespace chainerx {

class Array;

namespace internal {

class ArrayNode;

// This class is an internal data structure which holds array data/metadata (shape, dtype, ...) and backprop graph nodes and corresponding
// gradients.
class ArrayBody {
public:
    struct Params {
        Shape shape;
        Strides strides;
        Dtype dtype;
        Device& device;
        std::shared_ptr<void> data;
        int64_t offset;
    };

    // A backprop ID entry.
    // This is an aggregate data structure without logic.
    class BackpropEntry {
    public:
        explicit BackpropEntry(const BackpropId& backprop_id);

        const BackpropId& backprop_id() const { return backprop_id_; }

        bool is_grad_required() const { return is_grad_required_; }
        void SetIsGradRequired(bool value) { is_grad_required_ = value; }

        const std::shared_ptr<ArrayNode>& array_node() const { return array_node_; }
        bool has_array_node() const { return array_node_ != nullptr; }
        void SetArrayNode(std::shared_ptr<ArrayNode> array_node);

        std::unique_ptr<absl::optional<Array>>& grad() { return grad_; }
        const std::unique_ptr<absl::optional<Array>>& grad() const { return grad_; }
        void SetGrad(std::unique_ptr<absl::optional<Array>> grad);

    private:
        BackpropId backprop_id_;
        bool is_grad_required_{};
        std::shared_ptr<ArrayNode> array_node_{};
        std::unique_ptr<absl::optional<Array>> grad_;
    };

    ~ArrayBody() = default;

    ArrayBody(const ArrayBody&) = delete;
    ArrayBody(ArrayBody&&) = default;
    ArrayBody& operator=(const ArrayBody&) = delete;
    ArrayBody& operator=(ArrayBody&&) = delete;

    const Shape& shape() const { return shape_; }

    const Strides& strides() const { return strides_; }

    int8_t ndim() const { return shape_.ndim(); }

    Dtype dtype() const { return dtype_; }

    Device& device() const { return device_; }

    const std::shared_ptr<void>& data() const { return data_; }

    int64_t offset() const { return offset_; }

    const std::vector<BackpropEntry>& backprop_entries() const { return bps_; }

    bool has_backprop_entries() const { return !bps_.empty(); }

    std::vector<BackpropEntry>& backprop_entries() { return bps_; }

    int64_t GetItemSize() const { return chainerx::GetItemSize(dtype()); }

    bool IsContiguous() const { return internal::IsContiguous(shape(), strides(), GetItemSize()); }

    // Returns whether the backprop ID is registered in the array body.
    // It does not matter whether the graph is connected, i.e. whether the array body has an array node.
    // It does not matter whether the backprop ID is expired.
    // Backprop mode is not taken into account.
    bool HasBackpropId(const BackpropId& backprop_id) const {
        auto node = FindBackpropEntry(backprop_id);
        return node.has_value();
    }

    // Returns whether the array body has an array node corresponding to the backprop ID.
    // It does not matter whether the backprop ID is expired.
    bool HasArrayNode(const BackpropId& backprop_id) const {
        auto node = FindBackpropEntry(backprop_id);
        return node.has_value() && node->get().has_array_node();
    }

    // Returns whether the gradient of the specified backprop ID is marked as required.
    // It does not matter whether the graph is connected, i.e. whether the array body has an array node.
    // It does not matter whether the backprop ID is expired.
    // Backprop mode is not taken into account.
    bool IsGradRequired(const BackpropId& backprop_id) const {
        auto node = FindBackpropEntry(backprop_id);
        return node.has_value() && node->get().is_grad_required();
    }

    // Mark the gradient of the specified backprop ID as required.
    // Backprop mode is not taken into account.
    // If the backprop ID is not registered in the array body, it will be registered.
    // An array node will also be created, but only if the dtype kind is float.
    static void RequireGrad(const std::shared_ptr<ArrayBody>& body, const BackpropId& backprop_id) {
        CHAINERX_ASSERT(GetKind(body->dtype_) == DtypeKind::kFloat);

        bool create_array_node = GetKind(body->dtype_) == DtypeKind::kFloat;
        BackpropEntry& node = InstallBackpropId(body, backprop_id, create_array_node);
        node.SetIsGradRequired(true);
    }

    int64_t GetTotalSize() const { return shape().GetTotalSize(); }

    int64_t GetNBytes() const { return GetTotalSize() * GetItemSize(); }

    // Returns the array node corresponding to a given backprop ID.
    // Returns nullptr if it does not exist.
    const std::shared_ptr<ArrayNode>& GetArrayNode(const BackpropId& backprop_id) const {
        auto node = FindBackpropEntry(backprop_id);
        if (node.has_value()) {
            return node->get().array_node();
        }
        return kNullArrayNode;
    }

    // Creates a new backprop entry on the array body if it does not exist.
    // Reference to the newly-created or existing backward entry is returned.
    // The returned reference is only valid until the next call of InstallBackpropId on the same ArrayBody instance.
    // If `create_array_node` is true and there already is an array node, it does not create a new one.
    static BackpropEntry& InstallBackpropId(const std::shared_ptr<ArrayBody>& body, const BackpropId& backprop_id, bool create_array_node);

    // Creates a new backprop entry on the array body if it does not exist and sets the given array node.
    // Reference to the newly-created or existing backward entry is returned.
    // The returned reference is only valid until the next call of InstallBackpropId on the same ArrayBody instance.
    // If there already is an array node corresponding to the backprop ID, the behavior is undefined.
    static BackpropEntry& InstallBackpropId(
            const std::shared_ptr<ArrayBody>& body, const BackpropId& backprop_id, std::shared_ptr<ArrayNode> array_node);

    Params GetParams() const { return {shape_, strides_, dtype_, device_, data_, offset_}; }

    // Returns a gradient array.
    // Returns nullptr if the array does not belong to the specified graph.
    const absl::optional<Array>* GetGrad(const BackpropId& backprop_id) const;

    // Returns a gradient array.
    // Returns nullptr if the array does not belong to the specified graph.
    absl::optional<Array>* GetGrad(const BackpropId& backprop_id);

    // Sets a gradient array.
    // The behavior is undefined if there is no array node for the specified graph.
    void SetGrad(Array grad, const BackpropId& backprop_id);

    // Clears a gradient array.
    // The behavior is undefined if there is no array node for the specified graph.
    void ClearGrad(const BackpropId& backprop_id);

    // Returns the list of backprop IDs registered in the array body.
    std::vector<BackpropId> GetBackpropIds() const {
        std::vector<BackpropId> bps{};
        bps.reserve(bps_.size());
        for (const BackpropEntry& node : bps_) {
            bps.emplace_back(node.backprop_id());
        }
        return bps;
    }

private:
    friend std::shared_ptr<ArrayBody> CreateArrayBody(
            const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

    friend std::shared_ptr<ArrayBody> CreateArrayBody(Params params);

    ArrayBody(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

    explicit ArrayBody(Params params);

    // Asserts consistency of this instance.
    //
    // This function is no-op if CHAINERX_DEBUG is set.
    void AssertConsistency() const;

    // Adds an array node to the array body.
    // The array node must have been initialized with this array body in advance.
    // Otherwise the behavior is undefined.
    // It does nothing if an array node with the same backprop ID is already registered.
    // The returned reference is only valid until the next call of AddNode on this instance.
    static const std::shared_ptr<ArrayNode>& AddNode(
            const std::shared_ptr<ArrayBody>& body, ArrayBody::BackpropEntry& node, std::shared_ptr<ArrayNode> array_node);

    // Common implementation of FindBackpropEntry.
    template <typename ThisPtr, typename BackpropEntryType>
    static absl::optional<std::reference_wrapper<BackpropEntryType>> FindBackpropEntryImpl(
            ThisPtr this_ptr, const BackpropId& backprop_id) {
        for (BackpropEntryType& bp : this_ptr->bps_) {
            if (bp.backprop_id() == backprop_id) {
                return std::reference_wrapper<BackpropEntryType>{bp};
            }
        }
        return absl::nullopt;
    }

    // Finds the backprop entry corresponding to a given backprop ID and returns the reference to it.
    absl::optional<std::reference_wrapper<const BackpropEntry>> FindBackpropEntry(const BackpropId& backprop_id) const {
        return FindBackpropEntryImpl<const ArrayBody*, const BackpropEntry>(this, backprop_id);
    }

    // Finds the backprop entry corresponding to a given backprop ID and returns the reference to it.
    absl::optional<std::reference_wrapper<BackpropEntry>> FindBackpropEntry(const BackpropId& backprop_id) {
        return FindBackpropEntryImpl<ArrayBody*, BackpropEntry>(this, backprop_id);
    }

    // The use of non-POD static storage object here is safe, because destructing a shared_ptr with nullptr does not incur any
    // destruction order problem.
    static const std::shared_ptr<ArrayNode> kNullArrayNode;

    Shape shape_;
    Strides strides_;
    Dtype dtype_;
    Device& device_;
    std::shared_ptr<void> data_;
    int64_t offset_;  // in bytes

    std::vector<BackpropEntry> bps_;
};

std::shared_ptr<ArrayBody> CreateArrayBody(
        const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset);

std::shared_ptr<ArrayBody> CreateArrayBody(ArrayBody::Params params);

}  // namespace internal
}  // namespace chainerx
