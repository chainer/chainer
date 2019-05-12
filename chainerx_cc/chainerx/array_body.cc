#include "chainerx/array_body.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>

#include "chainerx/array.h"
#include "chainerx/array_body_leak_detection.h"
#include "chainerx/array_node.h"
#include "chainerx/backward.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace internal {

std::shared_ptr<ArrayBody> CreateArrayBody(
        const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset) {
    // Trick to use make_shared with private ctor
    struct ArrayBodyWithPublicCtor : ArrayBody {
        ArrayBodyWithPublicCtor(
                const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset)
            : ArrayBody{shape, strides, dtype, device, std::move(data), offset} {}
    };

    std::shared_ptr<ArrayBody> array_body =
            std::make_shared<ArrayBodyWithPublicCtor>(shape, strides, dtype, device, std::move(data), offset);

    if (internal::ArrayBodyLeakTracker* tracker = internal::ArrayBodyLeakDetectionScope::GetGlobalTracker()) {
        // TODO(niboshi): Make thread-safe
        (*tracker)(array_body);
    }

    return array_body;
}

std::shared_ptr<ArrayBody> CreateArrayBody(ArrayBody::Params params) {
    return CreateArrayBody(params.shape, params.strides, params.dtype, params.device, std::move(params.data), params.offset);
}

const std::shared_ptr<ArrayNode> ArrayBody::kNullArrayNode{nullptr};

ArrayBody::BackpropEntry::BackpropEntry(const BackpropId& backprop_id)
    : backprop_id_{backprop_id}, grad_{std::make_unique<nonstd::optional<Array>>(nonstd::nullopt)} {}

void ArrayBody::BackpropEntry::SetArrayNode(std::shared_ptr<ArrayNode> array_node) {
    CHAINERX_ASSERT(array_node->backprop_id() == backprop_id_);
    array_node_ = std::move(array_node);
}

void ArrayBody::BackpropEntry::SetGrad(std::unique_ptr<nonstd::optional<Array>> grad) { grad_ = std::move(grad); }

ArrayBody::ArrayBody(
        const Shape& shape,  // NOLINT(modernize-pass-by-value)
        const Strides& strides,  // NOLINT(modernize-pass-by-value)
        Dtype dtype,
        Device& device,
        std::shared_ptr<void> data,
        int64_t offset)
    : shape_{shape}, strides_{strides}, dtype_{dtype}, device_{device}, data_{std::move(data)}, offset_{offset} {}

ArrayBody::ArrayBody(Params params)
    : ArrayBody{params.shape, params.strides, params.dtype, params.device, std::move(params.data), params.offset} {}

const std::shared_ptr<ArrayNode>& ArrayBody::AddNode(
        const std::shared_ptr<ArrayBody>& body, ArrayBody::BackpropEntry& bp, std::shared_ptr<ArrayNode> array_node) {
    CHAINERX_ASSERT(bp.backprop_id() == array_node->backprop_id());

    body->AssertConsistency();

    // The body must be either unset (the array node is being created normally) or dead (the body is being replaced with a fabricated one,
    // as a retained output of backward)
    CHAINERX_ASSERT(array_node->weak_body().expired());

    // Do nothing and return the existing ArrayNode if found for this graph.
    if (bp.has_array_node()) {
        return bp.array_node();
    }

    // Connect the new backprop ID and the existing backprop IDs in this array body.
    for (const BackpropEntry& existing_bp : body->bps_) {
        if (existing_bp.has_array_node()) {
            Context& context = existing_bp.array_node()->device().context();
            context.ConnectBackpropIds(existing_bp.backprop_id(), array_node->backprop_id());
        }
    }

    internal::SetArrayNodeWeakBody(*array_node, body);

    // Assign the array node
    bp.SetArrayNode(std::move(array_node));

    body->AssertConsistency();
    return bp.array_node();
}

ArrayBody::BackpropEntry& ArrayBody::InstallBackpropId(
        const std::shared_ptr<ArrayBody>& body, const BackpropId& backprop_id, bool create_array_node) {
    CHAINERX_ASSERT(!create_array_node || GetKind(body->dtype_) == DtypeKind::kFloat);

    auto maybe_bp = body->FindBackpropEntry(backprop_id);
    if (!maybe_bp.has_value()) {
        body->bps_.emplace_back(backprop_id);
        maybe_bp = std::reference_wrapper<BackpropEntry>{body->bps_.back()};
    }

    BackpropEntry& bp = *maybe_bp;

    if (create_array_node) {
        if (!bp.has_array_node()) {
            std::shared_ptr<ArrayNode> array_node = std::make_shared<ArrayNode>(body->shape_, body->dtype_, body->device_, backprop_id);
            AddNode(body, bp, std::move(array_node));
        }

        CHAINERX_ASSERT(bp.has_array_node());
        CHAINERX_ASSERT(body->HasBackpropId(backprop_id));
    }

    return bp;
}

ArrayBody::BackpropEntry& ArrayBody::InstallBackpropId(
        const std::shared_ptr<ArrayBody>& body, const BackpropId& backprop_id, std::shared_ptr<ArrayNode> array_node) {
    CHAINERX_ASSERT(array_node != nullptr);
    CHAINERX_ASSERT(backprop_id == array_node->backprop_id());
    CHAINERX_ASSERT(GetKind(body->dtype_) == DtypeKind::kFloat);

    BackpropEntry& bp = InstallBackpropId(body, backprop_id, false);
    CHAINERX_ASSERT(!bp.has_array_node());  // Can't overwrite existing array node

    ArrayNode* array_node_ptr = array_node.get();

    AddNode(body, bp, std::move(array_node));

    CHAINERX_ASSERT(bp.array_node().get() == array_node_ptr);
    CHAINERX_ASSERT(body->HasBackpropId(backprop_id));
    return bp;
}

void ArrayBody::AssertConsistency() const {
    if (CHAINERX_DEBUG) {
        for (const BackpropEntry& bp : bps_) {
            // Assert uniqueness of backprop IDs
            for (const BackpropEntry& bp2 : bps_) {
                CHAINERX_ASSERT(&bp == &bp2 || bp.backprop_id() != bp2.backprop_id());
            }

            const std::shared_ptr<ArrayNode>& array_node = bp.array_node();
            CHAINERX_ASSERT(bp.grad() != nullptr);

            if (array_node != nullptr) {
                CHAINERX_ASSERT(bp.backprop_id() == array_node->backprop_id());
                CHAINERX_ASSERT(this == array_node->weak_body().lock().get());
                CHAINERX_ASSERT(bp.grad() != nullptr);

                // Array with integral dtypes cannot have array nodes.
                CHAINERX_ASSERT(GetKind(dtype()) == DtypeKind::kFloat);

                const nonstd::optional<Array>& grad = *bp.grad();

                if (grad.has_value()) {
                    CHAINERX_ASSERT(internal::GetArrayBody(*grad) != nullptr);
                    CHAINERX_ASSERT(array_node != nullptr);
                    CHAINERX_ASSERT(grad->shape() == array_node->shape());
                    CHAINERX_ASSERT(grad->dtype() == array_node->dtype());
                    CHAINERX_ASSERT(&grad->device() == &array_node->device());
                }
            }
        }
    }
}

void ArrayBody::SetGrad(Array grad, const BackpropId& backprop_id) {
    nonstd::optional<Array>* target_grad = GetGrad(backprop_id);
    CHAINERX_ASSERT(target_grad != nullptr);
    internal::SetGrad(*target_grad, std::move(grad), shape_, dtype_, device_);
}

void ArrayBody::ClearGrad(const BackpropId& backprop_id) {
    nonstd::optional<Array>* grad = GetGrad(backprop_id);
    CHAINERX_ASSERT(grad != nullptr);
    grad->reset();
}

const nonstd::optional<Array>* ArrayBody::GetGrad(const BackpropId& backprop_id) const {
    auto bp = FindBackpropEntry(backprop_id);
    if (!bp.has_value()) {
        return nullptr;
    }
    CHAINERX_ASSERT(bp->get().grad() != nullptr);
    return bp->get().grad().get();
}

nonstd::optional<Array>* ArrayBody::GetGrad(const BackpropId& backprop_id) {
    auto bp = FindBackpropEntry(backprop_id);
    if (!bp.has_value()) {
        return nullptr;
    }
    if (bp->get().grad() == nullptr) {
        return nullptr;
    }
    return bp->get().grad().get();
}

}  // namespace internal
}  // namespace chainerx
