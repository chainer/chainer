#include "xchainer/array.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array_body.h"
#include "xchainer/array_node.h"
#include "xchainer/array_repr.h"
#include "xchainer/backend.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/op_node.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace internal {

void SetUpOpNodes(
        const std::string& name,
        const std::vector<ConstArrayRef>& inputs,
        Array& out,
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

const std::shared_ptr<ArrayNode>& CreateArrayNode(Array& array, const GraphId& graph_id) {
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

size_t GetRequiredBytes(const Shape& shape, const Strides& strides, size_t element_size) {
    Expects(shape.ndim() == strides.ndim());
    if (shape.ndim() == 0) {
        return 0;
    }
    // Calculate the distance between the first and the last element, plus single element size.
    size_t total_bytes = element_size;
    for (int8_t i = 0; i < shape.ndim(); ++i) {
        total_bytes += (shape[i] - 1) * std::abs(strides[i]);
    }
    return total_bytes;
}

Array ArrayFromBuffer(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, const Strides& strides, Device& device) {
    auto bytesize = internal::GetRequiredBytes(shape, strides, GetElementSize(dtype));
    std::shared_ptr<void> device_data = device.FromBuffer(data, bytesize);
    return {shape, strides, dtype, device, device_data};
}

}  // namespace internal

Array Array::FromBuffer(const Shape& shape, Dtype dtype, const std::shared_ptr<void>& data, Device& device) {
    return internal::ArrayFromBuffer(shape, dtype, data, {shape, dtype}, device);
}

Array Array::Empty(const Shape& shape, Dtype dtype, Device& device) {
    auto bytesize = static_cast<size_t>(shape.GetTotalSize() * GetElementSize(dtype));
    std::shared_ptr<void> data = device.Allocate(bytesize);
    return {shape, Strides{shape, dtype}, dtype, device, data};
}

Array Array::Full(const Shape& shape, Scalar scalar, Dtype dtype, Device& device) {
    Array array = Empty(shape, dtype, device);
    array.Fill(scalar);
    return array;
}

Array Array::Full(const Shape& shape, Scalar scalar, Device& device) { return Full(shape, scalar, scalar.dtype(), device); }

Array Array::Zeros(const Shape& shape, Dtype dtype, Device& device) { return Full(shape, 0, dtype, device); }

Array Array::Ones(const Shape& shape, Dtype dtype, Device& device) { return Full(shape, 1, dtype, device); }

Array Array::EmptyLike(const Array& array, Device& device) { return Empty(array.shape(), array.dtype(), device); }

Array Array::FullLike(const Array& array, Scalar scalar, Device& device) { return Full(array.shape(), scalar, array.dtype(), device); }

Array Array::ZerosLike(const Array& array, Device& device) { return Zeros(array.shape(), array.dtype(), device); }

Array Array::OnesLike(const Array& array, Device& device) { return Ones(array.shape(), array.dtype(), device); }

Array::Array(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset)
    : body_(std::make_shared<internal::ArrayBody>(shape, strides, dtype, device, std::move(data), offset)) {}

Array::Array(const Array& other)
    : body_(std::make_shared<internal::ArrayBody>(
              other.shape(), other.strides(), other.dtype(), other.device(), other.body_->data_, other.offset(), other.body_->nodes_)) {}

Array& Array::operator+=(const Array& rhs) {
    Add(rhs, *this);
    return *this;
}

Array& Array::operator*=(const Array& rhs) {
    Mul(rhs, *this);
    return *this;
}

Array Array::operator+(const Array& rhs) const {
    Array out = Array::EmptyLike(*this, device());
    Add(rhs, out);
    return out;
}

Array Array::operator*(const Array& rhs) const {
    Array out = Array::EmptyLike(*this, device());
    Mul(rhs, out);
    return out;
}

Array Array::AddAt(const std::vector<ArrayIndex>& indices, const Array& addend) const {
    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), addend.dtype());

    Array out = this->AsConstant(CopyKind::kCopy);
    Array out_view = out.GetItem(indices);

    // TODO(sonots): broadcasting
    CheckEqual(out_view.shape(), addend.shape());

    device().Add(addend, out_view, out_view);

    auto this_backward_function = [](const Array& gout, const std::vector<GraphId>&) { return gout; };
    auto addend_backward_function = [indices](const Array& gout, const std::vector<GraphId>&) { return gout.GetItem(indices); };
    internal::SetUpOpNodes("add_at", {*this, addend}, out, {this_backward_function, addend_backward_function});

    return out;
}

Array Array::Transpose() const {
    Shape out_shape{shape().rbegin(), shape().rend()};
    Strides out_strides{strides().rbegin(), strides().rend()};
    Array out{out_shape, out_strides, dtype(), device(), body_->data_, offset()};
    internal::SetUpOpNodes("transpose", {*this}, out, {[](const Array& gout, const std::vector<GraphId>&) { return gout.Transpose(); }});
    return out;
}

Array Array::GetItem(const std::vector<ArrayIndex>& indices) const {
    std::vector<int64_t> out_shape;
    std::vector<int64_t> out_strides;
    int64_t out_offset = offset();
    int64_t i_in = 0;
    for (const ArrayIndex& index : indices) {
        switch (index.tag()) {
            case ArrayIndexTag::kSingleElement: {
                int64_t dim = shape()[i_in];
                if (index.index() < -dim || dim <= index.index()) {
                    throw DimensionError(
                            "Index " + std::to_string(index.index()) + " is out of bounds for axis " + std::to_string(i_in) +
                            " with size " + std::to_string(dim));
                }
                out_offset += strides()[i_in] * ((index.index() + dim) % dim);
                ++i_in;
                break;
            }
            case ArrayIndexTag::kSlice: {
                const Slice& slice = index.slice();
                int64_t slice_length = slice.GetLength(shape()[i_in]);
                out_offset += strides()[i_in] * slice.GetStart(shape()[i_in]);
                out_shape.push_back(slice_length);
                out_strides.push_back(strides()[i_in] * slice.step());
                ++i_in;
                break;
            }
            case ArrayIndexTag::kNewAxis:
                out_shape.push_back(1);
                out_strides.push_back(0);
                break;
            default:
                assert(false);
        }
    }
    for (int64_t i = i_in; i < ndim(); ++i) {
        out_shape.push_back(shape()[i]);
        out_strides.push_back(strides()[i]);
    }

    Array out{{out_shape.begin(), out_shape.end()}, {out_strides.begin(), out_strides.end()}, dtype(), device(), body_->data_, out_offset};

    auto backward_function = [ indices, other = *this ](const Array& gout, const std::vector<GraphId>&) {
        Array gin = Array::ZerosLike(other, other.device());
        return gin.AddAt(indices, gout);
    };
    internal::SetUpOpNodes("get_item", {*this}, out, {backward_function});

    return out;
}

Array Array::Reshape(const Shape& shape) const {
    const Shape& in_shape = this->shape();
    const Strides& in_strides = strides();

    // If the shape is unchanged, just return a view.
    if (in_shape == shape) {
        return *this;
    }

    // Check for invalid shape.
    int64_t total_size = in_shape.GetTotalSize();
    if (total_size != shape.GetTotalSize()) {
        throw DimensionError("Cannot reshape array of size " + std::to_string(total_size) + " into shape " + shape.ToString());
    }

    int64_t element_size = GetElementSize(dtype());
    Strides strides;
    if (total_size == 0) {
        // Calculate the strides for 0-sized array.
        std::vector<int64_t> rev_strides_vec;
        rev_strides_vec.push_back(element_size);
        for (int8_t i = shape.ndim() - 1; i >= 1; --i) {
            rev_strides_vec.push_back(rev_strides_vec.back() * std::max(int64_t{1}, shape[i]));
        }
        strides = Strides{rev_strides_vec.rbegin(), rev_strides_vec.rend()};
    } else {
        // Calculate the strides for non-0-sized array.

        // reduced_shape and reduced_strides are the shortest shape and strides which can be convertible from input shape and strides
        // without copy.
        std::vector<int64_t> reduced_shape;
        std::vector<int64_t> reduced_strides;
        if (in_shape.ndim() == 0) {
            // Input shape is (). Treat as if it were (1).
            reduced_shape.push_back(int64_t{1});
            reduced_strides.push_back(element_size);
        } else {
            // Add the first pair
            reduced_shape.reserve(in_shape.ndim());
            reduced_strides.reserve(in_shape.ndim());
            reduced_shape.push_back(in_shape[0]);
            reduced_strides.push_back(in_strides[0]);
            // Reduce the remaining
            for (int8_t i = 1; i < in_shape.ndim(); ++i) {
                int64_t dim = in_shape[i];
                int64_t st = in_strides[i];
                Expects(dim > 0);
                if (dim * st == reduced_strides.back()) {
                    // If the pair is compatible with the previous stride, reduce the pair to it.
                    reduced_shape.back() *= dim;
                    reduced_strides.back() = st;
                } else {
                    // Otherwise, add a new shape and stride.
                    reduced_shape.push_back(dim);
                    reduced_strides.push_back(st);
                }
            }
        }
        Ensures(reduced_shape.size() == reduced_strides.size());
        Ensures(!reduced_shape.empty());

        // Construct the strides for no-copy reshape.
        // If it's not possible, can_reshape_without_copy will be false.
        bool can_reshape_without_copy = true;
        std::vector<int64_t> strides_vec;
        if (shape.ndim() > 0) {
            int64_t last_stride = reduced_shape[0] * reduced_strides[0];
            size_t i_dim = 0;
            strides_vec.reserve(shape.ndim());
            for (int64_t dim : shape) {
                if (dim == 0) {
                    strides_vec.push_back(last_stride);
                    continue;
                }
                if (i_dim >= reduced_shape.size() || reduced_shape[i_dim] % dim != 0) {
                    strides_vec.clear();
                    can_reshape_without_copy = false;
                    break;
                }
                reduced_shape[i_dim] /= dim;
                last_stride = reduced_shape[i_dim] * reduced_strides[i_dim];
                strides_vec.push_back(last_stride);
                if (reduced_strides[i_dim] == 1) {
                    ++i_dim;
                }
            }
        }

        if (!can_reshape_without_copy) {
            // Reshape without copy is not possible.
            // TODO(niboshi): Implement it
            throw NotImplementedError("Reshape that requires a copy is not implemented yet.");
        }
        Ensures(strides_vec.size() == shape.size());

        strides = Strides{strides_vec.begin(), strides_vec.end()};
    }

    Array out{shape, strides, dtype(), device(), body_->data_, offset()};
    internal::SetUpOpNodes(
            "reshape", {*this}, out, {[in_shape](const Array& gout, const std::vector<GraphId>&) { return gout.Reshape(in_shape); }}, {});

    Ensures(out.shape() == shape);
    Ensures(out.strides().size() == shape.size());
    return out;
}

Array Array::BroadcastTo(const Shape& shape) const {
    const Shape& in_shape = this->shape();
    const Strides& in_strides = this->strides();

    if (in_shape.size() > shape.size()) {
        throw DimensionError("Cannot broadcast to smaller dimensions");
    }

    std::vector<int64_t> rev_strides;
    rev_strides.reserve(shape.size());
    int8_t i_in = in_shape.ndim() - 1;
    for (int8_t i_out = shape.ndim() - 1; i_out >= 0; --i_out) {
        int64_t out_dim = shape[i_out];
        // If this dimension is broadcasted, nonbroadcast_stride is unset.
        // Otherwise, it holds the new stride.
        nonstd::optional<int64_t> nonbroadcast_stride{};
        if (i_in >= 0) {
            int64_t in_dim = in_shape[i_in];
            int64_t in_stride = in_strides[i_in];
            --i_in;
            if (in_dim == 1 && in_stride == 0) {
                // do nothing; broadcast
            } else if (in_dim == out_dim) {
                nonbroadcast_stride = in_stride;
            } else {
                throw DimensionError("Invalid broadcast from " + in_shape.ToString() + " to " + shape.ToString());
            }
        } else {
            // do nothing; broadcast
        }

        if (nonbroadcast_stride.has_value()) {
            // non-broadcast dimension
            rev_strides.push_back(*nonbroadcast_stride);
        } else {
            // broadcast dimension
            if (out_dim == 1) {
                // broadcasted dimension is not broadcastable
                rev_strides.push_back(int64_t{1});
            } else {
                rev_strides.push_back(int64_t{0});
            }
        }
    }
    Ensures(rev_strides.size() == shape.size());

    return Array{shape, {rev_strides.rbegin(), rev_strides.rend()}, dtype(), device(), body_->data_, offset()};
}

Array Array::Copy() const {
    // No graph will be disconnected.
    Array out = AsConstant({}, CopyKind::kCopy);
    Ensures(out.IsContiguous());
    return out;
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

            Ensures(out.IsContiguous());
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

            Ensures(out.IsContiguous());
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

void Array::Fill(Scalar value) { device().Fill(*this, value); }

const nonstd::optional<Array>& Array::GetGrad(const GraphId& graph_id) const { return internal::GetArrayNode(*this, graph_id)->grad(); }

void Array::SetGrad(Array grad, const GraphId& graph_id) { internal::GetMutableArrayNode(*this, graph_id)->set_grad(std::move(grad)); }

void Array::ClearGrad(const GraphId& graph_id) { internal::GetMutableArrayNode(*this, graph_id)->ClearGrad(); }

std::string Array::ToString() const { return ArrayRepr(*this); }

void Array::Add(const Array& rhs, Array& out) const {
    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape(), rhs.shape());

    auto lhs_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array { return gout; };
    auto rhs_backward_function = lhs_backward_function;
    internal::SetUpOpNodes("add", {*this, rhs}, out, {lhs_backward_function, rhs_backward_function});

    device().Add(*this, rhs, out);
}

void Array::Mul(const Array& rhs, Array& out) const {
    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape(), rhs.shape());

    auto lhs_backward_function = [other = rhs](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        return gout * other.AsConstant(graph_ids_to_stop_gradient);
    };
    auto rhs_backward_function = [other = *this](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        return gout * other.AsConstant(graph_ids_to_stop_gradient);
    };
    internal::SetUpOpNodes("mul", {*this, rhs}, out, {lhs_backward_function, rhs_backward_function});

    device().Mul(*this, rhs, out);
}

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
