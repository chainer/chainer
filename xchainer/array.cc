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
#include "xchainer/error.h"
#include "xchainer/memory.h"
#include "xchainer/op_node.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace internal {

// Private definition of ArrayBody
ArrayBody::ArrayBody(const Shape& shape, Dtype dtype, bool is_contiguous, std::shared_ptr<void> data, int64_t offset)
    : shape_(shape), dtype_(dtype), is_contiguous_(is_contiguous), data_(std::move(data)), offset_(offset), nodes_() {}

ArrayBody::ArrayBody(const Shape& shape, Dtype dtype, bool is_contiguous, std::shared_ptr<void> data, int64_t offset,
                     std::vector<std::pair<GraphId, std::shared_ptr<ArrayNode>>> nodes)
    : shape_(shape), dtype_(dtype), is_contiguous_(is_contiguous), data_(std::move(data)), offset_(offset), nodes_(nodes) {}

bool ArrayBody::HasNode(const GraphId& graph_id) const {
    return std::find_if(nodes_.begin(), nodes_.end(), [&graph_id](const auto& graph_id_node) { return graph_id == graph_id_node.first; }) !=
           nodes_.end();
}

std::shared_ptr<const ArrayNode> ArrayBody::GetNode(const GraphId& graph_id) const {
    return std::const_pointer_cast<const ArrayNode>(GetMutableNode(graph_id));
}

const std::shared_ptr<ArrayNode>& ArrayBody::GetMutableNode(const GraphId& graph_id) const {
    auto it =
        std::find_if(nodes_.begin(), nodes_.end(), [&graph_id](const auto& graph_id_node) { return graph_id == graph_id_node.first; });
    if (it == nodes_.end()) {
        throw XchainerError("Cannot find ArrayNode for graph: " + graph_id);
    }
    return it->second;
}

const std::shared_ptr<ArrayNode>& ArrayBody::CreateNode(const GraphId& graph_id) {
    if (HasNode(graph_id)) {
        throw XchainerError("Duplicate graph registration: " + graph_id);
    }
    nodes_.emplace_back(graph_id, std::make_shared<ArrayNode>());
    return nodes_.back().second;
}

}  // namespace internal

Array::Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, bool is_contiguous, int64_t offset)
    : body_(std::make_shared<internal::ArrayBody>(shape, dtype, is_contiguous, std::move(data), offset)) {}

Array::Array(const Array& other)
    : body_(std::make_shared<internal::ArrayBody>(other.shape(), other.dtype(), other.is_contiguous(), other.body_->data_, other.offset(),
                                                  other.body_->nodes_)) {}

const nonstd::optional<Array>& Array::GetGrad(const GraphId& graph_id) const { return body_->GetNode(graph_id)->grad(); }

void Array::SetGrad(Array grad, const GraphId& graph_id) { body_->GetMutableNode(graph_id)->set_grad(std::move(grad)); }

void Array::ClearGrad(const GraphId& graph_id) { body_->GetMutableNode(graph_id)->ClearGrad(); }

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
    CreateGraph("copy", {*this}, out, {[](const Array& gout) { return gout; }});

    // TODO(hvy): When non-C-contiguous orders are supported, we cannot blindly copy all elements but need to take
    // is_contiguous_ and offset_ into account
    internal::MemoryCopy(out.data().get(), body_->data_.get(), total_bytes());
}

void Array::Add(const Array& rhs, Array& out) const {
    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape(), rhs.shape());

    auto lhs_backward_function = [](const Array& gout) -> Array { return gout; };
    auto rhs_backward_function = lhs_backward_function;
    CreateGraph("add", {*this, rhs}, out, {lhs_backward_function, rhs_backward_function});

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

    auto lhs_backward_function = [other_view = rhs](const Array& gout) { return gout * other_view; };
    auto rhs_backward_function = [other_view = *this](const Array& gout) { return gout * other_view; };
    CreateGraph("mul", {*this, rhs}, out, {lhs_backward_function, rhs_backward_function});

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

void CreateGraph(std::string name, std::vector<std::reference_wrapper<const Array>> inputs, Array& out,
                 std::vector<std::function<Array(const Array&)>> backward_functions) {
    std::unordered_map<GraphId, std::shared_ptr<OpNode>> graph_id_op_nodes;

    auto build_op_nodes = [&name, &graph_id_op_nodes](const GraphId& graph_id, const std::shared_ptr<ArrayNode>& next_node,
                                                      auto& backward_function) {
        auto& op_node = graph_id_op_nodes[graph_id];  // Create if not exists
        if (!op_node) {
            op_node = std::make_shared<OpNode>(name);
        }
        op_node->set_rank(std::max(op_node->rank(), next_node->rank()));
        op_node->RegisterNextNode(next_node);
        op_node->RegisterBackwardFunction(backward_function);
    };

    size_t nin = inputs.size();
    if (nin != backward_functions.size()) {
        throw XchainerError("Cannot currently construct a graph where numbers of input Arrays and backward functions do not match.");
    }
    for (size_t i = 0; i < nin; ++i) {
        for (auto& graph_id_node : inputs[i].get().nodes()) {  // For each graph
            build_op_nodes(graph_id_node.first, graph_id_node.second, backward_functions[i]);
        }
    }

    // Add OpNodes to output
    for (const auto& graph_id_op_node : graph_id_op_nodes) {
        const auto& graph_id = graph_id_op_node.first;
        const auto& op_node = graph_id_op_node.second;

        auto next_nodes = op_node->next_nodes();
        int64_t next_rank = (*std::max_element(next_nodes.begin(), next_nodes.end(), [](const auto& a, const auto& b) {
                                return a->rank() < b->rank();
                            }))->rank();

        auto& out_node = out.body()->CreateNode(graph_id);
        out_node->set_next_node(op_node);
        out_node->set_rank(next_rank + 1);
    }
}

void DebugDumpComputationalGraph(std::ostream& os, const Array& array, int indent) {
    for (const auto& graph_id_node : array.nodes()) {
        DebugDumpComputationalGraph(os, *graph_id_node.second, indent);
    }
}

}  // namespace xchainer
