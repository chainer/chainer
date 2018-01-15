#include "xchainer/array.h"

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

Array::Array(Shape shape, Dtype dtype, std::shared_ptr<void> data, std::shared_ptr<ArrayNode> node, bool requires_grad, bool is_contiguous,
             int64_t offset)
    : shape_(std::move(shape)),
      dtype_(dtype),
      data_(std::move(data)),
      requires_grad_(requires_grad),
      is_contiguous_(is_contiguous),
      offset_(offset),
      node_(std::move(node)) {}

Array::Array(const Array& other)
    : shape_(other.shape_),
      dtype_(other.dtype_),
      requires_grad_(other.requires_grad_),
      is_contiguous_(other.is_contiguous_),
      offset_(other.offset_),
      node_(std::make_shared<ArrayNode>()) {
    data_ = Allocate(GetCurrentDevice(), other.total_bytes());
    other.Copy(*this);
}

const std::shared_ptr<ArrayNode>& Array::RenewNode() {
    node_ = std::make_shared<ArrayNode>();
    return node_;
}

const nonstd::optional<Array>& Array::grad() const noexcept { return node_->grad(); }

void Array::set_grad(Array grad) { node_->set_grad(std::move(grad)); }

void Array::ClearGrad() noexcept { node_->ClearGrad(); }

Array Array::FromBuffer(const Shape& shape, Dtype dtype, std::shared_ptr<void> data) {
    auto bytesize = static_cast<size_t>(shape.total_size() * GetElementSize(dtype));
    std::shared_ptr<void> device_data = MemoryFromBuffer(GetCurrentDevice(), data, bytesize);
    return {shape, dtype, device_data, std::make_unique<ArrayNode>()};
}

Array Array::Empty(const Shape& shape, Dtype dtype) {
    auto bytesize = static_cast<size_t>(shape.total_size() * GetElementSize(dtype));
    std::shared_ptr<void> data = Allocate(GetCurrentDevice(), bytesize);
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
    requires_grad_ = requires_grad_ || rhs.requires_grad();
    return *this;
}

Array& Array::operator*=(const Array& rhs) {
    Mul(rhs, *this);
    requires_grad_ = requires_grad_ || rhs.requires_grad();
    return *this;
}

Array Array::operator+(const Array& rhs) const {
    Array out = Array::EmptyLike(*this);
    out.set_requires_grad(requires_grad_ || rhs.requires_grad());
    Add(rhs, out);
    return out;
}

Array Array::operator*(const Array& rhs) const {
    Array out = Array::EmptyLike(*this);
    out.set_requires_grad(requires_grad_ || rhs.requires_grad());
    Mul(rhs, out);
    return out;
}

void Array::Copy(Array& out) const {
    if (requires_grad_) {
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        auto in_func = [](const Array& gout) { return gout; };
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{in_func};
        std::shared_ptr<OpNode> op_node =
            std::make_shared<OpNode>("copy", std::vector<std::shared_ptr<const ArrayNode>>{node_}, backward_functions);
        out_node->set_next_node(op_node);
    }

    MemoryCopy(out.data().get(), data_.get(), total_bytes());
}

void Array::Add(const Array& rhs, Array& out) const {
    if ((&out == this || &out == &rhs) && out.requires_grad_) {
        throw XchainerError("In-place operation (Add) is not supported for an array with requires_grad=true.");
    }

    // TODO(sonots): dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape_, rhs.shape());

    if (requires_grad_ || rhs.requires_grad()) {
        const Array& lhs = *this;
        std::shared_ptr<const ArrayNode> lhs_node = node();
        std::shared_ptr<const ArrayNode> rhs_node = rhs.node();
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        std::function<Array(const Array&)> empty_func;
        auto lhs_func = lhs.requires_grad() ? [](const Array& gout) { return gout; } : empty_func;
        auto rhs_func = rhs.requires_grad() ? [](const Array& gout) { return gout; } : empty_func;
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
        std::shared_ptr<OpNode> op_node =
            std::make_shared<OpNode>("add", std::vector<std::shared_ptr<const ArrayNode>>{lhs_node, rhs_node}, backward_functions);
        out_node->set_next_node(op_node);
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
    if ((&out == this || &out == &rhs) && out.requires_grad_) {
        throw XchainerError("In-place operation (Mul) is not supported for an array with requires_grad=true.");
    }

    // TODO(sonots): dtype conversion
    CheckEqual(dtype_, rhs.dtype());
    // TODO(sonots): broadcasting
    CheckEqual(shape_, rhs.shape());

    if (requires_grad_ || rhs.requires_grad()) {
        std::shared_ptr<const ArrayNode> lhs_node = node();
        std::shared_ptr<const ArrayNode> rhs_node = rhs.node();
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        std::function<Array(const Array&)> empty_func;
        // TODO(sonots): turn off constructing graph (requires_grad) in backward (but, turn on for double backprop)
        // TODO(hvy): capture rhs by view (value)
        auto lhs_func = requires_grad_ ? [rhs](const Array& gout) { return gout * rhs; } : empty_func;
        auto rhs_func = empty_func;
        if (rhs.requires_grad()) {
            if (this == &out) {
                // deep copy for in-place operation to keep original input
                Array lhs = *this;
                rhs_func = [lhs](const Array& gout) { return gout * lhs; };
            } else {
                // TODO(takagi): capture lhs by view
                Array lhs = *this;
                rhs_func = [lhs](const Array& gout) { return gout * lhs; };
            }
        }
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
        std::shared_ptr<OpNode> op_node =
            std::make_shared<OpNode>("mul", std::vector<std::shared_ptr<const ArrayNode>>{lhs_node, rhs_node}, backward_functions);
        out_node->set_next_node(op_node);
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
