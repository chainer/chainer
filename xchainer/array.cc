#include "xchainer/array.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA

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

}  // namespace internal

Array::Array(const Shape& shape, Dtype dtype, std::shared_ptr<void> data, std::shared_ptr<ArrayNode> node, bool requires_grad,
             bool is_contiguous, int64_t offset, std::string graph_name)
    : body_(std::make_shared<internal::ArrayBody>(shape, dtype, requires_grad, is_contiguous, std::move(data), offset, std::move(node),
                                                  std::move(graph_name))) {}

Array::Array(const Array& other)
    : Array(other.shape(), other.dtype(), internal::Allocate(GetCurrentDevice(), other.total_bytes()), std::make_shared<ArrayNode>(),
            other.requires_grad(), true, 0) {
    // Memory layout-related members are not copied in this copy ctor since new C-contiguous memory is allocated
    other.CopyTo(*this);
}

const std::shared_ptr<ArrayNode>& Array::RenewNode() { return body_->node_ = std::make_shared<ArrayNode>(); }

const nonstd::optional<Array>& Array::grad() const noexcept { return body_->node_->grad(); }

void Array::set_grad(Array grad) { body_->node_->set_grad(std::move(grad)); }

void Array::ClearGrad() noexcept { body_->node_->ClearGrad(); }

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
    set_requires_grad(requires_grad() || rhs.requires_grad());
    return *this;
}

Array& Array::operator*=(const Array& rhs) {
    Mul(rhs, *this);
    set_requires_grad(requires_grad() || rhs.requires_grad());
    return *this;
}

Array Array::operator+(const Array& rhs) const {
    Array out = Array::EmptyLike(*this);
    out.set_requires_grad(requires_grad() || rhs.requires_grad());
    Add(rhs, out);
    return out;
}

Array Array::operator*(const Array& rhs) const {
    Array out = Array::EmptyLike(*this);
    out.set_requires_grad(requires_grad() || rhs.requires_grad());
    Mul(rhs, out);
    return out;
}

Array Array::Copy() const {
    Array out = Array::EmptyLike(*this);
    out.set_requires_grad(requires_grad());
    CopyTo(out);
    return out;
}

void Array::CopyTo(Array& out) const {
    if (requires_grad()) {
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        int64_t out_rank = node()->rank();
        auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{body_->node_};
        auto in_func = [](const Array& gout) { return gout; };
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{in_func};
        auto op_node = std::make_shared<OpNode>("copy", out_rank, next_nodes, backward_functions);
        out_node->set_next_node(op_node);
        out_node->set_rank(out_rank + 1);
    }

    // TODO(hvy): When non-C-contiguous orders are supported, we cannot blindly copy all elements but need to take
    // is_contiguous_ and offset_ into account
    internal::MemoryCopy(out.data().get(), body_->data_.get(), total_bytes());
}

void Array::Add(const Array& rhs, Array& out) const {
    if ((&out == this || &out == &rhs) && out.requires_grad()) {
        throw XchainerError("In-place operation (Add) is not supported for an array with requires_grad=true.");
    }

    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape(), rhs.shape());

    if (requires_grad() || rhs.requires_grad()) {
        const Array& lhs = *this;
        std::shared_ptr<ArrayNode> lhs_node = mutable_node();
        std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node();
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        int64_t out_rank = std::max(lhs_node->rank(), rhs_node->rank());
        auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{lhs_node, rhs_node};
        std::function<Array(const Array&)> empty_func;
        auto lhs_func = lhs.requires_grad() ? [](const Array& gout) { return gout; } : empty_func;
        auto rhs_func = rhs.requires_grad() ? [](const Array& gout) { return gout; } : empty_func;
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
        auto op_node = std::make_shared<OpNode>("add", out_rank, next_nodes, backward_functions);
        out_node->set_next_node(op_node);
        out_node->set_rank(out_rank + 1);
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
    if ((&out == this || &out == &rhs) && out.requires_grad()) {
        throw XchainerError("In-place operation (Mul) is not supported for an array with requires_grad=true.");
    }

    // TODO(sonots): dtype conversion
    CheckEqual(dtype(), rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape(), rhs.shape());

    if (requires_grad() || rhs.requires_grad()) {
        std::shared_ptr<ArrayNode> lhs_node = mutable_node();
        std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node();
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        int64_t out_rank = std::max(lhs_node->rank(), rhs_node->rank());
        auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{lhs_node, rhs_node};
        std::function<Array(const Array&)> empty_func;
        // TODO(sonots): turn off constructing graph (requires_grad) in backward (but, turn on for double backprop)
        auto lhs_func = requires_grad() ? [rhs_view = rhs.MakeView()](const Array& gout) { return gout * rhs_view; } : empty_func;
        auto rhs_func = rhs.requires_grad() ? [lhs_view = MakeView()](const Array& gout) { return gout * lhs_view; } : empty_func;
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
        auto op_node = std::make_shared<OpNode>("mul", out_rank, next_nodes, backward_functions);
        out_node->set_next_node(op_node);
        out_node->set_rank(out_rank + 1);
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
    DebugDumpComputationalGraph(os, *array.node(), indent);
}

}  // namespace xchainer
