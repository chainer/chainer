#include "xchainer/array.h"

#include <algorithm>
#include <cassert>
#include <cstring>
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
ArrayBody::ArrayBody(const Shape& shape, Dtype dtype, bool is_contiguous, std::shared_ptr<void> data, int64_t offset)
    : shape_(shape), dtype_(dtype), is_contiguous_(is_contiguous), data_(std::move(data)), offset_(offset), nodes_({}) {}

ArrayNode& ArrayBody::GetNode(const std::string& graph_name, bool requires_grad) {
    auto maybe_node_property = FindNodeProperty(graph_name);
    if (maybe_node_property) {
        maybe_node_property->requires_grad = requires_grad;  // TODO(hvy): Update/overwrite here or not?
        return *maybe_node_property->node;
    }
    auto node = std::make_shared<ArrayNode>();
    ArrayNodeGradientProperty node_property(node, requires_grad);
    nodes_.push_back({graph_name, node_property});
    return *nodes_.back().second.node;
}

}  // namespace internal

Array::Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, bool is_contiguous, int64_t offset)
    : body_(std::make_shared<internal::ArrayBody>(shape, dtype, is_contiguous, std::move(data), offset)) {}

Array::Array(const Array& other)
    : Array(other.shape(), other.dtype(), internal::Allocate(GetCurrentDevice(), other.total_bytes()), true, 0) {
    // Memory layout-related members are not copied in this copy ctor since new C-contiguous memory is allocated
    other.CopyTo(*this);
}

const nonstd::optional<Array>& Array::grad(const std::string& graph_name) const { return body_->node(graph_name)->grad(); }

void Array::set_grad(Array grad, const std::string& graph_name) { body_->mutable_node(graph_name)->set_grad(std::move(grad)); }

void Array::ClearGrad(const std::string& graph_name) { body_->mutable_node(graph_name)->ClearGrad(); }

Array Array::FromBuffer(const Shape& shape, Dtype dtype, std::shared_ptr<void> data) {
    auto bytesize = static_cast<size_t>(shape.total_size() * GetElementSize(dtype));
    std::shared_ptr<void> device_data = internal::MemoryFromBuffer(GetCurrentDevice(), data, bytesize);
    return {shape, dtype, device_data};
}

Array Array::Empty(const Shape& shape, Dtype dtype) {
    auto bytesize = static_cast<size_t>(shape.total_size() * GetElementSize(dtype));
    std::shared_ptr<void> data = internal::Allocate(GetCurrentDevice(), bytesize);
    return {shape, dtype, data};
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
    for (auto& named_node : body_->nodes_) {
        const auto& node_property = named_node.second;
        bool requires_grad = node_property.requires_grad;
        ArrayNode& out_node = out.body_->GetNode(named_node.first, requires_grad);
        if (requires_grad) {
            int64_t out_rank = node_property.node->rank();
            auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{node_property.node};
            auto in_func = [](const Array& gout) { return gout; };
            auto backward_functions = std::vector<std::function<Array(const Array&)>>{in_func};
            auto op_node = std::make_shared<OpNode>("copy", out_rank, next_nodes, backward_functions);
            out_node.set_next_node(op_node);
            out_node.set_rank(out_rank + 1);
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
        const std::string& graph_name = named_node.first;
        const auto& node_property = named_node.second;
        bool requires_grad = node_property.requires_grad;
        out.body_->GetNode(named_node.first, requires_grad);

        if (requires_grad) {
            std::shared_ptr<ArrayNode> lhs_node = mutable_node(graph_name);
            auto lhs_func = [](const Array& gout) { return gout; };
            OpNode& op_node = graph_op_nodes[graph_name];  // Create if not exists
            op_node.set_name("add");
            op_node.RegisterNextNode(lhs_node);
            op_node.RegisterBackwardFunction(lhs_func);
        }
    }

    // RHS
    for (auto& named_node : rhs.body_->nodes_) {
        const std::string& graph_name = named_node.first;
        const auto& node_property = named_node.second;
        bool requires_grad = node_property.requires_grad;
        out.body_->GetNode(named_node.first, requires_grad);

        if (requires_grad) {
            std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node(graph_name);
            auto rhs_func = [](const Array& gout) { return gout; };
            OpNode& op_node = graph_op_nodes[graph_name];  // Create if not exists
            op_node.set_name("add");
            op_node.RegisterNextNode(rhs_node);
            op_node.RegisterBackwardFunction(rhs_func);
        }
    }

    for (const auto& graph_op_node : graph_op_nodes) {
        // TODO(hvy): Better interface for creating a new node!
        ArrayNode& out_node = out.body_->GetNode(graph_op_node.first, true);
        gsl::span<const std::shared_ptr<ArrayNode>> next_nodes = graph_op_node.second.next_nodes();
        out_node.set_next_node(std::make_shared<OpNode>(graph_op_node.second));
        out_node.set_rank((*std::max_element(next_nodes.begin(), next_nodes.end(),
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
    for (auto& named_node : body_->nodes_) {  // For each graph
        const std::string& graph_name = named_node.first;
        const auto& node_property = named_node.second;
        bool requires_grad = node_property.requires_grad;
        out.body_->GetNode(graph_name, requires_grad);  // Create an ArrayNode internally if it does not exists for this graph

        if (requires_grad) {
            std::shared_ptr<ArrayNode> lhs_node = mutable_node(graph_name);
            auto lhs_func = [rhs_view = rhs.MakeView()](const Array& gout) { return gout * rhs_view; };
            OpNode& op_node = graph_op_nodes[graph_name];  // Create if not exists
            op_node.set_name("mul");
            op_node.RegisterNextNode(lhs_node);
            op_node.RegisterBackwardFunction(lhs_func);
        }
    }

    // RHS
    for (auto& named_node : rhs.body_->nodes_) {  // For each graph
        const std::string& graph_name = named_node.first;
        const auto& node_property = named_node.second;
        bool requires_grad = node_property.requires_grad;
        out.body_->GetNode(graph_name, requires_grad);

        if (requires_grad) {
            std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node(graph_name);
            auto rhs_func = [lhs_view = MakeView()](const Array& gout) { return gout * lhs_view; };
            OpNode& op_node = graph_op_nodes[graph_name];  // Create if not exists
            op_node.set_name("mul");
            op_node.RegisterNextNode(rhs_node);
            op_node.RegisterBackwardFunction(rhs_func);
        }
    }

    for (const auto& graph_op_node : graph_op_nodes) {
        ArrayNode& out_node = out.body_->GetNode(graph_op_node.first, true);
        // TODO(hvy): Fix intercface, how to initialize nodes, temp we are setting the gradient requirement to true here. Not so nice.
        gsl::span<const std::shared_ptr<ArrayNode>> next_nodes = graph_op_node.second.next_nodes();
        out_node.set_next_node(std::make_shared<OpNode>(graph_op_node.second));
        out_node.set_rank((*std::max_element(next_nodes.begin(), next_nodes.end(),
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
    for (auto& named_property : array.nodes()) {
        DebugDumpComputationalGraph(os, *named_property.second.node, indent);
    }
}

}  // namespace xchainer
