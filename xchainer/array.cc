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
    : shape_(shape), dtype_(dtype), is_contiguous_(is_contiguous), data_(std::move(data)), offset_(offset), nodes_() {}

bool ArrayBody::HasNode(const GraphId& graph_id) const {
    return std::find_if(nodes_.begin(), nodes_.end(), [&graph_id](const auto& graph_id_node) { return graph_id == graph_id_node.first; }) !=
           nodes_.end();
}

std::shared_ptr<const ArrayNode> ArrayBody::Node(const GraphId& graph_id) const {
    auto it =
        std::find_if(nodes_.begin(), nodes_.end(), [&graph_id](const auto& graph_id_node) { return graph_id == graph_id_node.first; });
    if (it == nodes_.end()) {
        throw XchainerError("Cannot find ArrayNode for graph: " + graph_id);
    }
    return it->second;
}

const std::shared_ptr<ArrayNode>& ArrayBody::MutableNode(const GraphId& graph_id) const {
    auto it =
        std::find_if(nodes_.begin(), nodes_.end(), [&graph_id](const auto& graph_id_node) { return graph_id == graph_id_node.first; });
    if (it == nodes_.end()) {
        throw XchainerError("Cannot find ArrayNode for graph: " + graph_id);
    }
    return it->second;
}

const std::shared_ptr<ArrayNode>& ArrayBody::CreateNode(const GraphId& graph_id) {
    if (HasNode(graph_id)) {
        throw XchainerError("Duplicate graph registrationh: " + graph_id);
    }
    nodes_.push_back({graph_id, std::make_shared<ArrayNode>()});
    return nodes_.back().second;
}

}  // namespace internal

Array::Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, bool is_contiguous, int64_t offset)
    : body_(std::make_shared<internal::ArrayBody>(shape, dtype, is_contiguous, std::move(data), offset)) {}

Array::Array(const Array& other)
    : Array(other.shape(), other.dtype(), internal::Allocate(GetCurrentDevice(), other.total_bytes()), true, 0) {
    // Memory layout-related members are not copied in this copy ctor since new C-contiguous memory is allocated
    other.CopyTo(*this);
}

const Array& Array::Grad(const GraphId& graph_id) const {
    auto& grad = body_->Node(graph_id)->grad();
    if (!grad) {
        throw XchainerError("Gradient not set for graph: " + graph_id);
    }
    return *grad;
}

void Array::SetGrad(Array grad, const GraphId& graph_id) { body_->MutableNode(graph_id)->set_grad(std::move(grad)); }

void Array::ClearGrad(const GraphId& graph_id) { body_->MutableNode(graph_id)->ClearGrad(); }

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
    for (const auto& graph_id_node : body_->nodes_) {
        const auto& graph_id = graph_id_node.first;
        const auto& next_node = graph_id_node.second;

        auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{next_node};
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{[](const Array& gout) { return gout; }};
        int64_t prev_rank = next_node->rank();
        auto op_node = std::make_shared<OpNode>("copy", prev_rank, next_nodes, backward_functions);

        auto& out_node = out.body_->CreateNode(graph_id);
        out_node->set_next_node(op_node);
        out_node->set_rank(prev_rank + 1);
    }

    // TODO(hvy): When non-C-contiguous orders are supported, we cannot blindly copy all elements but need to take
    // is_contiguous_ and offset_ into account
    internal::MemoryCopy(out.data().get(), body_->data_.get(), total_bytes());
}

void Array::Add(const Array& rhs, Array& out) const {
    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape(), rhs.shape());

    std::unordered_map<std::string, OpNode> graph_id_op_nodes;

    auto add_op = [&out, &graph_id_op_nodes](auto& graph_id_node) {
        const auto& graph_id = graph_id_node.first;
        const auto& next_node = graph_id_node.second;
        auto backward_function = [](const Array& gout) { return gout; };
        OpNode& op_node = graph_id_op_nodes[graph_id];  // Create if not exists
        op_node.set_name("add");
        op_node.set_rank(std::max(op_node.rank(), next_node->rank()));
        op_node.RegisterNextNode(next_node);
        op_node.RegisterBackwardFunction(backward_function);
    };

    // Create OpNodes
    for (auto& named_node : body_->nodes_) {  // For each graph
        add_op(named_node);
    }
    for (auto& named_node : rhs.body_->nodes_) {  // For each graph
        add_op(named_node);
    }

    // Add OpNodes to output
    for (const auto& graph_id_op_node : graph_id_op_nodes) {
        const auto& graph_id = graph_id_op_node.first;
        const auto& op_node = graph_id_op_node.second;
        auto& out_node = out.body_->CreateNode(graph_id);
        gsl::span<const std::shared_ptr<ArrayNode>> next_nodes = op_node.next_nodes();
        out_node->set_next_node(std::make_shared<OpNode>(op_node));
        out_node->set_rank(
            (*std::max_element(next_nodes.begin(), next_nodes.end(), [](const auto& a, const auto& b) { return a->rank() < b->rank(); }))
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
    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape(), rhs.shape());

    std::unordered_map<std::string, OpNode> graph_id_op_nodes;

    auto add_op = [&out, &graph_id_op_nodes](auto& graph_id_node, const Array& other) {
        const auto& graph_id = graph_id_node.first;
        const auto& next_node = graph_id_node.second;
        auto backward_function = [other_view = other.MakeView()](const Array& gout) { return gout * other_view; };
        OpNode& op_node = graph_id_op_nodes[graph_id];  // Create if not exists
        op_node.set_name("mul");
        op_node.set_rank(std::max(op_node.rank(), next_node->rank()));
        op_node.RegisterNextNode(next_node);
        op_node.RegisterBackwardFunction(backward_function);
    };

    // Create OpNodes
    for (auto& named_node : body_->nodes_) {  // For each graph
        add_op(named_node, rhs);
    }
    for (auto& named_node : rhs.body_->nodes_) {  // For each graph
        add_op(named_node, *this);
    }

    // Add OpNodes to output
    for (const auto& graph_id_op_node : graph_id_op_nodes) {
        const auto& graph_id = graph_id_op_node.first;
        const auto& op_node = graph_id_op_node.second;
        auto& out_node = out.body_->CreateNode(graph_id);
        gsl::span<const std::shared_ptr<ArrayNode>> next_nodes = op_node.next_nodes();
        out_node->set_next_node(std::make_shared<OpNode>(op_node));
        out_node->set_rank(
            (*std::max_element(next_nodes.begin(), next_nodes.end(), [](const auto& a, const auto& b) { return a->rank() < b->rank(); }))
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
    for (const auto& graph_id_node : array.nodes()) {
        DebugDumpComputationalGraph(os, *graph_id_node.second, indent);
    }
}

}  // namespace xchainer
