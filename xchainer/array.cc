#include "xchainer/array.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>  // TODO(hvy): delete me
#include <string>
#include <unordered_map>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gsl/gsl>

#include "xchainer/array_fill.h"
#include "xchainer/array_math.h"
#include "xchainer/array_repr.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/array_fill.h"
#include "xchainer/cuda/array_math.h"
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/memory.h"
#include "xchainer/op_node.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace internal {

// Private definition of ArrayBody
ArrayBody::ArrayBody(const Shape& shape, Dtype dtype, bool requires_grad, bool is_contiguous, std::shared_ptr<void> data, int64_t offset,
                     std::shared_ptr<ArrayNode> node, std::string graph_name)
    : shape_(shape),
      dtype_(dtype),
      is_contiguous_(is_contiguous),
      data_(std::move(data)),
      offset_(offset),
      nodes_({{std::move(graph_name), {std::move(node), requires_grad}}}) {}

const std::shared_ptr<ArrayNode>& ArrayBody::GetOrCreateNode(const std::string& graph_name, bool default_requires_grad) {
    auto named_nodes =
        std::find_if(nodes_.begin(), nodes_.end(),
                     [&graph_name](const std::pair<std::string, ArrayNodeGradientProperty>& node) { return node.first == graph_name; });

    if (named_nodes != nodes_.end()) {
        // Return match
        return named_nodes->second.node;
    }

    // Create and return new
    std::shared_ptr<ArrayNode> node = std::make_shared<ArrayNode>();
    ArrayNodeGradientProperty node_property(node, default_requires_grad);
    nodes_.push_back({graph_name, node_property});
    return nodes_.back().second.node;
}

}  // namespace internal

Array::Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, std::shared_ptr<ArrayNode> node, bool requires_grad,
             bool is_contiguous, int64_t offset, const std::string& graph_name)
    : body_(std::make_shared<internal::ArrayBody>(shape, dtype, requires_grad, is_contiguous, std::move(data), offset, std::move(node),
                                                  std::move(graph_name))) {}

Array::Array(const Array& other)
    : Array(other.shape(), other.dtype(), internal::Allocate(GetCurrentDevice(), other.total_bytes()), std::make_shared<ArrayNode>(),
            other.requires_grad(), true, 0) {
    // Memory layout-related members are not copied in this copy ctor since new C-contiguous memory is allocated
    other.CopyTo(*this);
}

// TODO(hvy): Multi-graph support
// const std::shared_ptr<ArrayNode>& Array::RenewNode() { return body_->node_ = std::make_shared<ArrayNode>(); }

const nonstd::optional<Array>& Array::grad(const std::string& graph_name) const { return body_->node(graph_name)->grad(); }

void Array::set_grad(Array grad, const std::string& graph_name) { body_->mutable_node(graph_name)->set_grad(std::move(grad)); }

void Array::ClearGrad(const std::string& graph_name) { body_->mutable_node(graph_name)->ClearGrad(); }

// TODO(hvy): Multi-graph support, i.e. specify graph_name
Array Array::FromBuffer(const Shape& shape, Dtype dtype, std::shared_ptr<void> data) {
    auto bytesize = static_cast<size_t>(shape.total_size() * GetElementSize(dtype));
    std::shared_ptr<void> device_data = internal::MemoryFromBuffer(GetCurrentDevice(), data, bytesize);
    return {shape, dtype, device_data, std::make_unique<ArrayNode>()};
}

Array Array::Empty(const Shape& shape, Dtype dtype) {
    auto bytesize = static_cast<size_t>(shape.total_size() * GetElementSize(dtype));
    std::shared_ptr<void> data = internal::Allocate(GetCurrentDevice(), bytesize);
    return {shape, dtype, data, std::make_unique<ArrayNode>()};
}

Array Array::Full(const Shape& shape, Scalar scalar, Dtype dtype) {
    Array array = Empty(shape, dtype);
    array.Fill(scalar);
    return array;
}

Array Array::Full(const Shape& shape, Scalar scalar) { return Full(shape, scalar, scalar.dtype()); }

Array Array::Zeros(const Shape& shape, Dtype dtype) { return Full(shape, 0, dtype); }

Array Array::Ones(const Shape& shape, Dtype dtype) { return Full(shape, 1, dtype); }

Array Array::EmptyLike(const Array& array) { return Empty(array.shape(), array.dtype()); }

Array Array::FullLike(const Array& array, Scalar scalar) { return Full(array.shape(), scalar, array.dtype()); }

Array Array::ZerosLike(const Array& array) { return Zeros(array.shape(), array.dtype()); }

Array Array::OnesLike(const Array& array) { return Ones(array.shape(), array.dtype()); }

Array& Array::operator+=(const Array& rhs) {
    Add(rhs, *this);
    return *this;
}

Array& Array::operator*=(const Array& rhs) {
    Mul(rhs, *this);
    return *this;
}

Array Array::operator+(const Array& rhs) const {
    Array out = Array::EmptyLike(*this);
    Add(rhs, out);
    return out;
}

Array Array::operator*(const Array& rhs) const {
    Array out = Array::EmptyLike(*this);
    Mul(rhs, out);
    return out;
}

Array Array::Copy() const {
    Array out = Array::EmptyLike(*this);
    CopyTo(out);
    return out;
}

void Array::CopyTo(Array& out) const {
    for (auto& named_node : body_->nodes_) {  // For each graph, or ArrayNode
        const std::string& graph_name = named_node.first;
        if (named_node.second.requires_grad) {  // If ArrayNode requires gradients
            const ArrayNodeGradientProperty& node_prop = named_node.second;
            std::shared_ptr<ArrayNode> out_node = out.GetOrCreateNode(graph_name, true);
            int64_t out_rank = node_prop.node->rank();
            auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{node_prop.node};
            auto in_func = [](const Array& gout) { return gout; };
            auto backward_functions = std::vector<std::function<Array(const Array&)>>{in_func};
            auto op_node = std::make_shared<OpNode>("copy", out_rank, next_nodes, backward_functions);
            out_node->set_next_node(op_node);
            out_node->set_rank(out_rank + 1);
            // out.set_requires_grad(true, graph_name);
        }
    }

    // TODO(hvy): When non-C-contiguous orders are supported, we cannot blindly copy all elements but need to take
    // is_contiguous_ and offset_ into account
    internal::MemoryCopy(out.data().get(), body_->data_.get(), total_bytes());
}

void Array::Add(const Array& rhs, Array& out) const {
    /*
    if ((&out == this || &out == &rhs) && out.requires_grad()) {
        throw XchainerError("In-place operation (Add) is not supported for an array with requires_grad=true.");
    }
    */

    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape(), rhs.shape());

    std::unordered_map<std::string, OpNode> graph_op_nodes;

    // LHS
    for (auto& named_node : body_->nodes_) {  // For each graph
        std::cout << "LHS ADD" << named_node.second.requires_grad << std::endl;
        const std::string& graph_name = named_node.first;
        if (named_node.second.requires_grad) {
            std::shared_ptr<ArrayNode> lhs_node = mutable_node(graph_name);
            auto lhs_func = [](const Array& gout) { return gout; };
            OpNode& op_node = graph_op_nodes[graph_name];  // Create if not exists
            op_node.set_name("add");
            std::cout << "REGISTER LHS ADD" << std::endl;
            op_node.RegisterNextNode(lhs_node);
            op_node.RegisterBackwardFunction(lhs_func);
            // out.set_requires_grad(true, graph_name);
        }
    }

    // RHS
    for (auto& named_node : rhs.nodes()) {
        std::cout << "RHS ADD" << named_node.second.requires_grad << std::endl;
        const std::string& graph_name = named_node.first;
        if (named_node.second.requires_grad) {
            std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node(graph_name);
            auto rhs_func = [](const Array& gout) { return gout; };
            OpNode& op_node = graph_op_nodes[graph_name];
            op_node.set_name("add");
            std::cout << "REGISTER RHS ADD" << std::endl;
            op_node.RegisterNextNode(rhs_node);
            op_node.RegisterBackwardFunction(rhs_func);
        }
    }

    std::cout << "ADD FIN LHS, RHS" << std::endl;

    for (const auto& graph_op_node : graph_op_nodes) {
        // TODO(hvy): Better interface for creating a new node!
        std::shared_ptr<ArrayNode> out_node = out.GetOrCreateNode(graph_op_node.first, true);
        out.set_requires_grad(true, "");
        std::cout << "OUT : " << out.requires_grad("") << std::endl;
        gsl::span<const std::shared_ptr<ArrayNode>> next_nodes = graph_op_node.second.next_nodes();
        out_node->set_next_node(std::make_shared<OpNode>(graph_op_node.second));
        out_node->set_rank((*std::max_element(next_nodes.begin(), next_nodes.end(),
                                              [](const std::shared_ptr<ArrayNode>& a, const std::shared_ptr<ArrayNode>& b) {
                                                  return a->rank() < b->rank();
                                              }))
                               ->rank() +
                           1);
    }

    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        xchainer::Add(*this, rhs, out);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        xchainer::cuda::Add(*this, rhs, out);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

void Array::Mul(const Array& rhs, Array& out) const {
    /*
    if ((&out == this || &out == &rhs) && out.requires_grad()) {
        throw XchainerError("In-place operation (Mul) is not supported for an array with requires_grad=true.");
    }
    */

    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape(), rhs.shape());

    std::unordered_map<std::string, OpNode> graph_op_nodes;

    // LHS
    for (auto& named_node : body_->nodes_) {
        const std::string& graph_name = named_node.first;
        if (named_node.second.requires_grad) {
            std::shared_ptr<ArrayNode> lhs_node = mutable_node(graph_name);
            // TODO(sonots): turn off constructing graph (requires_grad) in backward (but, turn on for double backprop)
            auto lhs_func = [rhs_view = rhs.MakeView()](const Array& gout) { return gout * rhs_view; };
            OpNode& op_node = graph_op_nodes[graph_name];
            op_node.set_name("mul");
            op_node.RegisterNextNode(lhs_node);
            op_node.RegisterBackwardFunction(lhs_func);
        }
    }

    // RHS
    for (auto& named_node : rhs.nodes()) {
        const std::string& graph_name = named_node.first;
        if (named_node.second.requires_grad) {
            std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node(graph_name);
            auto rhs_func = [lhs_view = MakeView()](const Array& gout) { return gout * lhs_view; };
            OpNode& op_node = graph_op_nodes[graph_name];
            op_node.set_name("mul");
            op_node.RegisterNextNode(rhs_node);
            op_node.RegisterBackwardFunction(rhs_func);
        }
    }

    for (const auto& graph_op_node : graph_op_nodes) {
        std::shared_ptr<ArrayNode> out_node = out.GetOrCreateNode(graph_op_node.first, true);
        // TODO(hvy): Fix intercface, how to initialize nodes, temp we are setting the gradient requirement to true here. Not so nice.
        out.set_requires_grad(true, "");
        gsl::span<const std::shared_ptr<ArrayNode>> next_nodes = graph_op_node.second.next_nodes();
        out_node->set_next_node(std::make_shared<OpNode>(graph_op_node.second));
        out_node->set_rank((*std::max_element(next_nodes.begin(), next_nodes.end(),
                                              [](const std::shared_ptr<ArrayNode>& a, const std::shared_ptr<ArrayNode>& b) {
                                                  return a->rank() < b->rank();
                                              }))
                               ->rank() +
                           1);
    }

    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        xchainer::Mul(*this, rhs, out);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        xchainer::cuda::Mul(*this, rhs, out);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

void Array::Fill(Scalar value) {
    Device device = GetCurrentDevice();
    if (device == MakeDevice("cpu")) {
        xchainer::Fill(*this, value);
#ifdef XCHAINER_ENABLE_CUDA
    } else if (device == MakeDevice("cuda")) {
        xchainer::cuda::Fill(*this, value);
#endif  // XCHAINER_ENABLE_CUDA
    } else {
        throw DeviceError("invalid device");
    }
}

std::string Array::ToString() const { return ArrayRepr(*this); }

namespace {

void DebugDumpComputationalGraph(std::ostream& os, const ArrayNode& array_node, int indent) {
    static const char kIndentChar = ' ';

    os << std::string(static_cast<size_t>(indent * 2), kIndentChar) << "ArrayNode<" << &array_node << ">" << std::endl;

    std::shared_ptr<const OpNode> op = array_node.next_node();
    if (op) {
        os << std::string(static_cast<size_t>((indent + 1) * 2), kIndentChar) << "Op<" << op->name() << ">" << std::endl;
        for (const std::shared_ptr<const ArrayNode>& next_node : op->next_nodes()) {
            DebugDumpComputationalGraph(os, *next_node, static_cast<size_t>(indent + 2));
        }
    }
}

}  // namespace

void DebugDumpComputationalGraph(std::ostream& os, const Array& array, int indent) {
    // DebugDumpComputationalGraph(os, *array.node(), indent);
}

}  // namespace xchainer
