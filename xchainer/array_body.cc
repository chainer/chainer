#include "xchainer/array_body.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/backward.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace internal {

ArrayBody::ArrayBody(Shape shape, Strides strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset)
    : shape_{std::move(shape)}, strides_{std::move(strides)}, dtype_{dtype}, device_{device}, data_{std::move(data)}, offset_{offset} {}

ArrayBody::ArrayBody(Params params)
    : ArrayBody{params.shape, params.strides, params.dtype, params.device, std::move(params.data), params.offset} {}

const std::shared_ptr<ArrayNode>& ArrayBody::AddNode(std::shared_ptr<ArrayNode> array_node) {
    AssertConsistency();
    assert(this == array_node->GetBody().get());
    auto it = std::find_if(nodes_.begin(), nodes_.end(), [&array_node](const std::shared_ptr<ArrayNode>& existing_node) {
        return existing_node->graph_id() == array_node->graph_id();
    });
    if (it != nodes_.end()) {
        return *it;  // Do nothing and return the existing ArrayNode if found for this graph.
    }

    nodes_.emplace_back(std::move(array_node));
    grads_.emplace_back(std::make_unique<nonstd::optional<Array>>(nonstd::nullopt));

    AssertConsistency();
    return nodes_.back();
}

void ArrayBody::AssertConsistency() const {
#ifndef NDEBUG
    assert(nodes_.size() == grads_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const std::shared_ptr<ArrayNode>& array_node = nodes_[i];
        const nonstd::optional<Array>& grad = *grads_[i];
        assert(array_node != nullptr);
        assert(this == array_node->GetBody().get());
        if (grad.has_value()) {
            assert(grad->body() != nullptr);
            assert(grad->shape() == array_node->shape());
            assert(grad->dtype() == array_node->dtype());
            assert(&grad->device() == &array_node->device());
        }
    }
#endif  // NDEBUG
}

nonstd::optional<size_t> ArrayBody::GetNodeIndex(const GraphId& graph_id) const {
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i]->graph_id() == graph_id) {
            return i;
        }
    }
    return nonstd::nullopt;
}

void ArrayBody::SetGrad(Array grad, const GraphId& graph_id) {
    nonstd::optional<Array>* target_grad = GetGrad(graph_id);
    assert(target_grad != nullptr);
    internal::SetGrad(*target_grad, std::move(grad), shape_, dtype_, device_);
}

void ArrayBody::AccumulateGrad(Array partial_grad, const GraphId& graph_id) {
    nonstd::optional<Array>* target_grad = GetGrad(graph_id);
    assert(target_grad != nullptr);
    internal::AccumulateGrad(*target_grad, std::move(partial_grad), shape_, dtype_, device_);
}

void ArrayBody::ClearGrad(const GraphId& graph_id) {
    nonstd::optional<Array>* grad = GetGrad(graph_id);
    if (grad == nullptr) {
        throw XchainerError{"Array does not belong to the graph: '", graph_id, "'."};
    }
    grad->reset();
}

template <typename ThisPtr, typename ReturnType>
ReturnType ArrayBody::GetGradImpl(ThisPtr this_ptr, const GraphId& graph_id) {
    nonstd::optional<size_t> i = this_ptr->GetNodeIndex(graph_id);
    if (!i.has_value()) {
        return nullptr;
    }
    assert(*i < this_ptr->grads_.size());
    return this_ptr->grads_[*i].get();
}

template nonstd::optional<Array>* ArrayBody::GetGradImpl<ArrayBody*, nonstd::optional<Array>*>(ArrayBody*, const GraphId&);
template const nonstd::optional<Array>* ArrayBody::GetGradImpl<const ArrayBody*, const nonstd::optional<Array>*>(
        const ArrayBody*, const GraphId&);

}  // namespace internal
}  // namespace xchainer
