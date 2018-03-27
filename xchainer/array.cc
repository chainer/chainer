#include "xchainer/array.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>

#include <nonstd/optional.hpp>

#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/array_repr.h"
#include "xchainer/backend.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/op_node.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/indexing.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/routines/math.h"
#include "xchainer/routines/util.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace internal {

Array MakeArray(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset) {
    return Array{shape, strides, dtype, device, std::move(data), offset};
}

void SetUpOpNodes(
        const std::string& name,
        const std::vector<ConstArrayRef>& inputs,
        const Array& out,
        const std::vector<std::function<Array(const Array&, const std::vector<GraphId>&)>>& backward_functions,
        const std::vector<GraphId>& graph_ids_to_stop_gradients) {
    if (inputs.size() != backward_functions.size()) {
        throw XchainerError("Cannot construct a graph where numbers of input Arrays and backward functions do not match.");
    }

    std::unordered_map<GraphId, std::shared_ptr<OpNode>> graph_edges;

    // Helper function to create an edge in the graph
    auto create_edge = [&name, &graph_edges](const std::shared_ptr<ArrayNode>& next_node, auto& backward_function) {
        std::shared_ptr<OpNode>& op_node = graph_edges[next_node->graph_id()];  // Create if not exists
        if (!op_node) {
            op_node = std::make_shared<OpNode>(name);
        }
        op_node->set_rank(std::max(op_node->rank(), next_node->rank()));
        op_node->RegisterNextNode(next_node, backward_function);
    };

    for (size_t i = 0; i < inputs.size(); ++i) {                                  // For each input
        for (const std::shared_ptr<ArrayNode>& node : inputs[i].get().nodes()) {  // For each graph, create an edge
            if (find(graph_ids_to_stop_gradients.begin(), graph_ids_to_stop_gradients.end(), node->graph_id()) ==
                graph_ids_to_stop_gradients.end()) {
                create_edge(node, backward_functions[i]);
            }
        }
    }

    if (!graph_edges.empty() && std::any_of(inputs.begin(), inputs.end(), [&out](const Array& input) { return &out == &input; })) {
        throw XchainerError("In-place operation (" + name + ") is not supported for an array that require gradients.");
    }

    // Bind edges to output
    for (const auto& edge : graph_edges) {
        const GraphId& graph_id = edge.first;
        const std::shared_ptr<OpNode>& op_node = edge.second;

        const std::shared_ptr<ArrayNode>& out_node = CreateArrayNode(out, graph_id);
        out_node->set_next_node(op_node);
        out_node->set_rank(op_node->rank() + 1);
    }
}

bool HasArrayNode(const Array& array, const GraphId& graph_id) {
    return std::find_if(array.nodes().begin(), array.nodes().end(), [&graph_id](const auto& node) {
               return graph_id == node->graph_id();
           }) != array.nodes().end();
}

const std::shared_ptr<ArrayNode>& CreateArrayNode(const Array& array, const GraphId& graph_id) {
    if (HasArrayNode(array, graph_id)) {
        throw XchainerError("Duplicate graph registration: '" + graph_id + "'.");
    }
    array.nodes().emplace_back(std::make_shared<ArrayNode>(graph_id));
    return array.nodes().back();
}

std::shared_ptr<const ArrayNode> GetArrayNode(const Array& array, const GraphId& graph_id) { return GetMutableArrayNode(array, graph_id); }

const std::shared_ptr<ArrayNode>& GetMutableArrayNode(const Array& array, const GraphId& graph_id) {
    auto it = std::find_if(
            array.nodes().begin(), array.nodes().end(), [&graph_id](const auto& node) { return graph_id == node->graph_id(); });
    if (it == array.nodes().end()) {
        throw XchainerError("Array does not belong to the graph: '" + graph_id + "'.");
    }
    return *it;
}

}  // namespace internal

Array Array::FromBuffer(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device) {
    return routines::FromBuffer(shape, dtype, data, device);
}

Array Array::Empty(const Shape& shape, Dtype dtype, Device& device) { return routines::Empty(shape, dtype, device); }

Array Array::Full(const Shape& shape, Scalar scalar, Dtype dtype, Device& device) { return routines::Full(shape, scalar, dtype, device); }

Array Array::Full(const Shape& shape, Scalar scalar, Device& device) { return routines::Full(shape, scalar, device); }

Array Array::Zeros(const Shape& shape, Dtype dtype, Device& device) { return routines::Zeros(shape, dtype, device); }

Array Array::Ones(const Shape& shape, Dtype dtype, Device& device) { return routines::Ones(shape, dtype, device); }

Array Array::EmptyLike(const Array& array, Device& device) { return routines::EmptyLike(array, device); }

Array Array::FullLike(const Array& array, Scalar scalar, Device& device) { return routines::FullLike(array, scalar, device); }

Array Array::ZerosLike(const Array& array, Device& device) { return routines::ZerosLike(array, device); }

Array Array::OnesLike(const Array& array, Device& device) { return routines::OnesLike(array, device); }

Array::Array(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset)
    : body_(std::make_shared<internal::ArrayBody>(shape, strides, dtype, device, std::move(data), offset)) {}

Array& Array::operator+=(const Array& rhs) { return routines::IAdd(*this, rhs); }

const Array& Array::operator+=(const Array& rhs) const { return routines::IAdd(*this, rhs); }

Array Array::operator+(const Array& rhs) const { return routines::Add(*this, rhs); }

Array& Array::operator*=(const Array& rhs) { return routines::IMul(*this, rhs); }

const Array& Array::operator*=(const Array& rhs) const { return routines::IMul(*this, rhs); }

Array Array::operator*(const Array& rhs) const { return routines::Mul(*this, rhs); }

Array Array::At(const std::vector<ArrayIndex>& indices) const { return routines::At(*this, indices); }

Array Array::Transpose() const { return routines::Transpose(*this); }

Array Array::Reshape(const Shape& shape) const { return routines::Reshape(*this, shape); }

Array Array::Squeeze(const nonstd::optional<std::vector<int8_t>>& axis) const { return routines::Squeeze(*this, axis); }

Array Array::BroadcastTo(const Shape& shape) const { return routines::BroadcastTo(*this, shape); }

Array Array::Sum(const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) const {
    std::vector<int8_t> sorted_axis;
    if (axis.has_value()) {
        sorted_axis = routines::internal::GetSortedAxes(*axis, shape().ndim());
    } else {
        // Fill with all axes
        sorted_axis.resize(shape().size());
        std::iota(sorted_axis.begin(), sorted_axis.end(), int8_t{0});
    }

    // Calculate output shape
    std::vector<int64_t> out_shape_vec;
    out_shape_vec.reserve(shape().ndim());
    std::vector<int8_t> out_axis;
    int8_t i_axis = 0;
    for (int8_t i = 0; i < shape().ndim(); ++i) {
        if (i_axis < static_cast<int8_t>(sorted_axis.size()) && i == sorted_axis[i_axis]) {
            ++i_axis;
            if (keepdims) {
                out_shape_vec.push_back(int64_t{1});
            }
        } else {
            out_shape_vec.push_back(shape()[i]);
            out_axis.push_back(i);
        }
    }

    Array out = Array::Empty({out_shape_vec.begin(), out_shape_vec.end()}, dtype(), device());
    if (keepdims) {
        // Set reduced strides of the output array to 0
        assert(out.IsContiguous());
        std::vector<int64_t> out_strides_vec(out.strides().begin(), out.strides().end());
        for (int8_t i_axis : sorted_axis) {
            out_strides_vec[i_axis] = 0;
        }
        out.body_->strides_ = Strides{out_strides_vec.begin(), out_strides_vec.end()};
    }
    device().Sum(*this, sorted_axis, out);

    auto backward_function = [ sorted_axis, in_shape = shape(), keepdims ](const Array& gout, const std::vector<GraphId>&) {
        assert(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));

        if (!(in_shape.ndim() == 0 || sorted_axis.empty() || keepdims)) {
            std::vector<int64_t> out_shape_broadcastable{gout.shape().begin(), gout.shape().end()};
            for (auto axis : sorted_axis) {
                out_shape_broadcastable.insert(out_shape_broadcastable.begin() + axis, 1);
            }
            return gout.Reshape({out_shape_broadcastable.begin(), out_shape_broadcastable.end()}).BroadcastTo(in_shape);
        }
        return gout.BroadcastTo(in_shape);
    };
    internal::SetUpOpNodes("sum", {*this}, out, {backward_function});

    return out;
}

Array Array::Copy() const {
    // No graph will be disconnected.
    Array out = AsConstant({}, CopyKind::kCopy);
    assert(out.IsContiguous());
    return out;
}

Array Array::MakeView() const {
    return Array{std::make_shared<internal::ArrayBody>(shape(), strides(), dtype(), device(), body_->data_, offset(), body_->nodes_)};
}

Array Array::ToDevice(Device& dst_device) const {
    Device& src_device = body_->device_;
    nonstd::optional<Array> out;

    if (&src_device == &dst_device) {
        // Return an alias.
        out.emplace(AsConstant(CopyKind::kView));
    } else {
        // Make a contiguous copy, then transfer it to the destination device.
        Array src_contig = AsConstant(CopyKind::kCopy);

        std::shared_ptr<void> dst_data;
        if (src_device.backend().SupportsTransfer(src_device, dst_device)) {
            // Use src backend for transfer.
            dst_data = src_device.TransferDataTo(dst_device, src_contig.data(), src_contig.offset(), src_contig.GetTotalBytes());
        } else if (dst_device.backend().SupportsTransfer(src_device, dst_device)) {
            // Use dst backend for transfer.
            dst_data = dst_device.TransferDataFrom(src_device, src_contig.data(), src_contig.offset(), src_contig.GetTotalBytes());
        } else {
            // Neither backends support transfer.
            throw XchainerError(
                    "Transfer between devices is not supported: src='" + src_device.name() + "' dst='" + dst_device.name() + "'.");
        }
        out.emplace(
                Array{src_contig.shape(), src_contig.strides(), src_contig.dtype(), dst_device, std::move(dst_data), src_contig.offset()});
    }

    assert(out.has_value());

    // Connect the graph.
    // Backward operation is implemented as backward-transfer.
    internal::SetUpOpNodes(
            "transfer",
            {*this},
            *out,
            {[&src_device](const Array& gout, const std::vector<GraphId>&) -> Array { return gout.ToDevice(src_device); }},
            {});
    return std::move(*out);
}

Array Array::AsConstant(CopyKind kind) const {
    switch (kind) {
        case CopyKind::kCopy: {
            Array out = Array::EmptyLike(*this, device());
            device().Copy(*this, out);

            assert(out.IsContiguous());
            return std::move(out);
        }
        case CopyKind::kView:
            return Array{shape(), strides(), dtype(), device(), body_->data_, offset()};
        default:
            assert(false);  // should never be reached
    }
}

Array Array::AsConstant(const std::vector<GraphId>& graph_ids, CopyKind kind) const {
    switch (kind) {
        case CopyKind::kCopy: {
            Array out = Array::EmptyLike(*this, device());
            internal::SetUpOpNodes("copy", {*this}, out, {[](const Array& gout, const std::vector<GraphId>&) { return gout; }}, graph_ids);
            device().Copy(*this, out);

            assert(out.IsContiguous());
            return std::move(out);
        }
        case CopyKind::kView: {
            Array out{shape(), strides(), dtype(), device(), body_->data_, offset()};

            // Duplicate the array nodes only when graph IDs are not found in specified graph_ids.
            for (const std::shared_ptr<ArrayNode>& node : nodes()) {
                if (std::find(graph_ids.begin(), graph_ids.end(), node->graph_id()) == graph_ids.end()) {
                    // extend the graph
                    out.body_->nodes_.emplace_back(node);
                }
            }
            return std::move(out);
        }
        default:
            assert(false);  // should never be reached
    }
}

void Array::Fill(Scalar value) const { device().Fill(*this, value); }

const nonstd::optional<Array>& Array::GetGrad(const GraphId& graph_id) const { return internal::GetArrayNode(*this, graph_id)->grad(); }

void Array::SetGrad(Array grad, const GraphId& graph_id) const {
    internal::GetMutableArrayNode(*this, graph_id)->set_grad(std::move(grad));
}

void Array::ClearGrad(const GraphId& graph_id) const { internal::GetMutableArrayNode(*this, graph_id)->ClearGrad(); }

std::string Array::ToString() const { return ArrayRepr(*this); }

namespace {

void DebugDumpComputationalGraph(std::ostream& os, const ArrayNode& array_node, int indent) {
    static const char kIndentChar = ' ';

    os << std::string(static_cast<size_t>(indent * 2), kIndentChar) << "ArrayNode<" << &array_node << ">" << std::endl;

    std::shared_ptr<const OpNode> op = array_node.next_node();
    if (op) {
        os << std::string(static_cast<size_t>((indent + 1) * 2), kIndentChar) << "Op<" << op->name() << "," << op.get() << ">" << std::endl;
        for (const std::shared_ptr<const ArrayNode>& next_node : op->next_nodes()) {
            DebugDumpComputationalGraph(os, *next_node, static_cast<size_t>(indent + 2));
        }
    }
}

}  // namespace

void DebugDumpComputationalGraph(std::ostream& os, const Array& array, const GraphId& graph_id, int indent) {
    DebugDumpComputationalGraph(os, *internal::GetArrayNode(array, graph_id), indent);
}

}  // namespace xchainer
