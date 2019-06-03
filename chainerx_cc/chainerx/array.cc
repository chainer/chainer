#include "chainerx/array.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "chainerx/array_body.h"
#include "chainerx/array_node.h"
#include "chainerx/array_repr.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/op_node.h"
#include "chainerx/routines/binary.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/reduction.h"
#include "chainerx/routines/routines_util.h"
#include "chainerx/routines/sorting.h"
#include "chainerx/routines/statistics.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace internal {

BackpropId GetArrayBackpropId(const Array& array, const nonstd::optional<BackpropId>& backprop_id) {
    return backprop_id.has_value() ? *backprop_id : array.device().context().default_backprop_id();
}

Array MakeArray(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset) {
    return Array{shape, strides, dtype, device, std::move(data), offset};
}

std::vector<std::shared_ptr<ArrayBody>> MoveArrayBodies(std::vector<Array>&& arrays) {
    std::vector<std::shared_ptr<ArrayBody>> array_body_ptrs;
    array_body_ptrs.reserve(arrays.size());
    for (Array& array : arrays) {
        array_body_ptrs.emplace_back(MoveArrayBody(std::move(array)));
    }
    return array_body_ptrs;
}

std::vector<std::shared_ptr<ArrayBody>> MoveArrayBodies(std::vector<nonstd::optional<Array>>&& arrays) {
    std::vector<std::shared_ptr<ArrayBody>> array_body_ptrs;
    array_body_ptrs.reserve(arrays.size());
    for (nonstd::optional<Array>& array : arrays) {
        if (array.has_value()) {
            array_body_ptrs.emplace_back(MoveArrayBody(std::move(*array)));
        } else {
            array_body_ptrs.emplace_back(nullptr);
        }
    }
    return array_body_ptrs;
}

}  // namespace internal

Array::Array(const Shape& shape, const Strides& strides, Dtype dtype, Device& device, std::shared_ptr<void> data, int64_t offset)
    : body_{internal::CreateArrayBody(shape, strides, dtype, device, std::move(data), offset)} {}

Array Array::operator-() const { return Negative(*this); }

Array Array::operator==(const Array& rhs) const { return Equal(*this, rhs); }

Array Array::operator!=(const Array& rhs) const { return NotEqual(*this, rhs); }

Array Array::operator>(const Array& rhs) const { return Greater(*this, rhs); }

Array Array::operator>=(const Array& rhs) const { return GreaterEqual(*this, rhs); }

Array Array::operator<(const Array& rhs) const { return Less(*this, rhs); }

Array Array::operator<=(const Array& rhs) const { return LessEqual(*this, rhs); }

Array& Array::operator+=(const Array& rhs) {
    internal::IAdd(*this, rhs);
    return *this;
}

Array& Array::operator+=(Scalar rhs) {
    internal::IAdd(*this, rhs);
    return *this;
}

Array& Array::operator-=(const Array& rhs) {
    internal::ISubtract(*this, rhs);
    return *this;
}

Array& Array::operator-=(Scalar rhs) {
    internal::ISubtract(*this, rhs);
    return *this;
}

Array& Array::operator*=(const Array& rhs) {
    internal::IMultiply(*this, rhs);
    return *this;
}

Array& Array::operator*=(Scalar rhs) {
    internal::IMultiply(*this, rhs);
    return *this;
}

Array& Array::operator/=(const Array& rhs) {
    internal::IDivide(*this, rhs);
    return *this;
}

Array& Array::operator/=(Scalar rhs) {
    internal::IDivide(*this, rhs);
    return *this;
}

Array& Array::operator&=(const Array& rhs) {
    internal::IBitwiseAnd(*this, rhs);
    return *this;
}

Array& Array::operator&=(Scalar rhs) {
    internal::IBitwiseAnd(*this, rhs);
    return *this;
}

Array& Array::operator|=(const Array& rhs) {
    internal::IBitwiseOr(*this, rhs);
    return *this;
}

Array& Array::operator|=(Scalar rhs) {
    internal::IBitwiseOr(*this, rhs);
    return *this;
}

Array& Array::operator^=(const Array& rhs) {
    internal::IBitwiseXor(*this, rhs);
    return *this;
}

Array& Array::operator^=(Scalar rhs) {
    internal::IBitwiseXor(*this, rhs);
    return *this;
}

const Array& Array::operator+=(const Array& rhs) const {
    internal::IAdd(*this, rhs);
    return *this;
}

const Array& Array::operator+=(Scalar rhs) const {
    internal::IAdd(*this, rhs);
    return *this;
}

const Array& Array::operator-=(const Array& rhs) const {
    internal::ISubtract(*this, rhs);
    return *this;
}

const Array& Array::operator-=(Scalar rhs) const {
    internal::ISubtract(*this, rhs);
    return *this;
}

const Array& Array::operator*=(const Array& rhs) const {
    internal::IMultiply(*this, rhs);
    return *this;
}

const Array& Array::operator*=(Scalar rhs) const {
    internal::IMultiply(*this, rhs);
    return *this;
}

const Array& Array::operator/=(const Array& rhs) const {
    internal::IDivide(*this, rhs);
    return *this;
}

const Array& Array::operator/=(Scalar rhs) const {
    internal::IDivide(*this, rhs);
    return *this;
}

const Array& Array::operator&=(const Array& rhs) const {
    internal::IBitwiseAnd(*this, rhs);
    return *this;
}

const Array& Array::operator&=(Scalar rhs) const {
    internal::IBitwiseAnd(*this, rhs);
    return *this;
}

const Array& Array::operator|=(const Array& rhs) const {
    internal::IBitwiseOr(*this, rhs);
    return *this;
}

const Array& Array::operator|=(Scalar rhs) const {
    internal::IBitwiseOr(*this, rhs);
    return *this;
}

const Array& Array::operator^=(const Array& rhs) const {
    internal::IBitwiseXor(*this, rhs);
    return *this;
}

const Array& Array::operator^=(Scalar rhs) const {
    internal::IBitwiseXor(*this, rhs);
    return *this;
}

Array Array::operator+(const Array& rhs) const { return chainerx::Add(*this, rhs); }

Array Array::operator+(Scalar rhs) const { return chainerx::Add(*this, rhs); }

Array Array::operator-(const Array& rhs) const { return chainerx::Subtract(*this, rhs); }

Array Array::operator-(Scalar rhs) const { return chainerx::Subtract(*this, rhs); }

Array Array::operator*(const Array& rhs) const { return Multiply(*this, rhs); }

Array Array::operator*(Scalar rhs) const { return Multiply(*this, rhs); }

Array Array::operator/(const Array& rhs) const { return chainerx::Divide(*this, rhs); }

Array Array::operator/(Scalar rhs) const { return chainerx::Divide(*this, rhs); }

Array Array::operator&(const Array& rhs) const { return chainerx::BitwiseAnd(*this, rhs); }

Array Array::operator&(Scalar rhs) const { return chainerx::BitwiseAnd(*this, rhs); }

Array Array::operator|(const Array& rhs) const { return chainerx::BitwiseOr(*this, rhs); }

Array Array::operator|(Scalar rhs) const { return chainerx::BitwiseOr(*this, rhs); }

Array Array::operator^(const Array& rhs) const { return chainerx::BitwiseXor(*this, rhs); }

Array Array::operator^(Scalar rhs) const { return chainerx::BitwiseXor(*this, rhs); }

Array Array::At(const std::vector<ArrayIndex>& indices) const { return internal::At(*this, indices); }

Array Array::Transpose(const OptionalAxes& axes) const { return chainerx::Transpose(*this, axes); }

Array Array::Reshape(const Shape& newshape) const { return chainerx::Reshape(*this, newshape); }

Array Array::Squeeze(const OptionalAxes& axis) const { return chainerx::Squeeze(*this, axis); }

Array Array::Swapaxes(int8_t axis1, int8_t axis2) const { return chainerx::Swapaxes(*this, axis1, axis2); }

Array Array::BroadcastTo(const Shape& shape) const { return chainerx::BroadcastTo(*this, shape); }

Array Array::ArgMax(const OptionalAxes& axis) const { return chainerx::ArgMax(*this, axis); }

Array Array::ArgMin(const OptionalAxes& axis) const { return chainerx::ArgMin(*this, axis); }

Array Array::Sum(const OptionalAxes& axis, bool keepdims) const { return chainerx::Sum(*this, axis, keepdims); }

Array Array::Max(const OptionalAxes& axis, bool keepdims) const { return chainerx::AMax(*this, axis, keepdims); }

Array Array::Min(const OptionalAxes& axis, bool keepdims) const { return chainerx::AMin(*this, axis, keepdims); }

Array Array::Mean(const OptionalAxes& axis, bool keepdims) const { return chainerx::Mean(*this, axis, keepdims); }

Array Array::Var(const OptionalAxes& axis, bool keepdims) const { return chainerx::Var(*this, axis, keepdims); }

Array Array::All(const OptionalAxes& axis, bool keepdims) const { return chainerx::All(*this, axis, keepdims); }

Array Array::Any(const OptionalAxes& axis, bool keepdims) const { return chainerx::Any(*this, axis, keepdims); }

Array Array::Dot(const Array& b) const { return chainerx::Dot(*this, b); }

Array Array::Take(const Array& indices, int8_t axis) const { return chainerx::Take(*this, indices, axis); }

Array Array::Copy() const { return chainerx::Copy(*this); }

Array Array::MakeView() const {
    Array out{shape(), strides(), dtype(), device(), data(), offset()};

    BackwardBuilder bb{"view", *this, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([](BackwardContext& bctx) { bctx.input_grad() = *bctx.output_grad(); });
    }
    bb.Finalize();

    return out;
}

Array Array::ToDevice(Device& dst_device) const {
    Device& src_device = body_->device();
    Array out;

    // TODO(sonots): Avoid copying data between native devices, e.g., from native:0 to native:1 for performance.
    if (&src_device == &dst_device) {
        // Return an alias.
        out = AsGradStopped(CopyKind::kView);
    } else {
        // Make a contiguous copy to transfer it to the destination device.
        Array src_contig = AsContiguous(AsGradStopped(CopyKind::kView));

        std::shared_ptr<void> dst_data;
        if (src_device.backend().SupportsTransfer(src_device, dst_device)) {
            // Use src backend for transfer.
            dst_data = src_device.TransferDataTo(dst_device, src_contig.data(), src_contig.offset(), src_contig.GetNBytes());
        } else if (dst_device.backend().SupportsTransfer(src_device, dst_device)) {
            // Use dst backend for transfer.
            dst_data = dst_device.TransferDataFrom(src_device, src_contig.data(), src_contig.offset(), src_contig.GetNBytes());
        } else {
            // Neither backends support transfer.
            throw ChainerxError{"Transfer between devices is not supported: src='", src_device.name(), "' dst='", dst_device.name(), "'."};
        }
        out = Array{src_contig.shape(), src_contig.strides(), src_contig.dtype(), dst_device, std::move(dst_data)};
    }

    CHAINERX_ASSERT(internal::GetArrayBody(out) != nullptr);

    // Backward operation is implemented as backward-transfer.
    BackwardBuilder bb{"transfer", *this, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([&src_device](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad()->ToDevice(src_device); });
    }
    bb.Finalize();

    // TODO(niboshi): This assertion must succeed but currently it does not because AsContiguousArray reshapes {} to {1}.
    // CHAINERX_ASSERT(out.shape() == shape());
    CHAINERX_ASSERT(out.dtype() == dtype());
    return out;
}

Array Array::ToNative() const {
    Context& context = device().backend().context();
    Device& native_device = context.GetNativeBackend().GetDevice(0);
    return ToDevice(native_device);
}

namespace {

Array CopyOrMakeView(const Array& array, CopyKind kind) {
    switch (kind) {
        case CopyKind::kCopy:
            return array.Copy();
        case CopyKind::kView:
            return array.MakeView();
        default:
            CHAINERX_NEVER_REACH();
    }
}

}  // namespace

Array Array::AsGradStopped(CopyKind kind) const {
    NoBackpropModeScope scope{device().context()};
    return CopyOrMakeView(*this, kind);
}

Array Array::AsGradStopped(gsl::span<const BackpropId> backprop_ids, CopyKind kind) const {
    NoBackpropModeScope scope{std::vector<BackpropId>{backprop_ids.begin(), backprop_ids.end()}};
    return CopyOrMakeView(*this, kind);
}

Array Array::AsType(Dtype dtype, bool copy) const {
    Dtype src_dtype = this->dtype();
    if (!copy && dtype == src_dtype) {
        return *this;
    }

    Array out = Empty(shape(), dtype, device());
    device().backend().CallKernel<AsTypeKernel>(*this, out);

    if (GetKind(dtype) == DtypeKind::kFloat) {
        BackwardBuilder bb{"astype", *this, out};
        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([src_dtype](BackwardContext& bctx) { bctx.input_grad() = bctx.output_grad()->AsType(src_dtype); });
        }
        bb.Finalize();
    }

    CHAINERX_ASSERT(out.IsContiguous());
    return out;
}

void Array::Fill(Scalar value) const {
    internal::CheckNoUnsafeInplace(*this, {});
    device().backend().CallKernel<FillKernel>(*this, value);
}

const nonstd::optional<Array>& Array::GetGrad(const nonstd::optional<BackpropId>& backprop_id) const {
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(*this, backprop_id);
    if (!IsGradRequired(actual_backprop_id)) {
        throw ChainerxError{"Array is not flagged as requiring gradient for backprop id: '", actual_backprop_id, "'."};
    }
    const nonstd::optional<Array>* grad = body_->GetGrad(actual_backprop_id);
    CHAINERX_ASSERT(grad != nullptr);
    return *grad;
}

void Array::SetGrad(Array grad, const nonstd::optional<BackpropId>& backprop_id) const {
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(*this, backprop_id);
    nonstd::optional<Array>* target_grad = body_->GetGrad(actual_backprop_id);
    if (target_grad == nullptr) {
        throw ChainerxError{"Array is constant with respect to the computation for backprop ID: '", actual_backprop_id, "'."};
    }

    // Setting the gradient flags the array to require gradient, so that it can return the gradient with GetGrad().
    RequireGrad(actual_backprop_id);

    internal::SetGrad(*target_grad, std::move(grad), shape(), dtype(), device());
}

void Array::ClearGrad(const nonstd::optional<BackpropId>& backprop_id) const {
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(*this, backprop_id);
    if (!body_->HasArrayNode(actual_backprop_id)) {
        throw ChainerxError{"Array is constant with respect to the computation for backprop ID: '", actual_backprop_id, "'."};
    }
    body_->ClearGrad(actual_backprop_id);
}

bool Array::IsBackpropRequired(const nonstd::optional<BackpropId>& backprop_id) const {
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(*this, backprop_id);
    return body_->HasArrayNode(actual_backprop_id) && chainerx::IsBackpropRequired(actual_backprop_id);
}

bool Array::IsBackpropRequired(AnyGraph /*any_graph*/) const {
    const std::vector<std::shared_ptr<internal::ArrayNode>>& array_nodes = body_->nodes();
    return std::any_of(array_nodes.begin(), array_nodes.end(), [](const std::shared_ptr<const internal::ArrayNode>& array_node) {
        return chainerx::IsBackpropRequired(array_node->backprop_id());
    });
}

bool Array::IsGradRequired(const nonstd::optional<BackpropId>& backprop_id) const {
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(*this, backprop_id);
    return body_->IsGradRequired(actual_backprop_id);
}

template <typename T>
T& Array::RequireGradImpl(T& array, const nonstd::optional<BackpropId>& backprop_id) {
    if (GetKind(array.dtype()) != DtypeKind::kFloat) {
        throw DtypeError{"Array with integral dtype (", GetDtypeName(array.dtype()), ") cannot compute gradient"};
    }
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(array, backprop_id);
    internal::ArrayBody::RequireGrad(internal::GetArrayBody(array), actual_backprop_id);
    return array;
}

template const Array& Array::RequireGradImpl<const Array>(const Array& array, const nonstd::optional<BackpropId>& backprop_id);
template Array& Array::RequireGradImpl<Array>(Array& array, const nonstd::optional<BackpropId>& backprop_id);

std::string Array::ToString() const { return ArrayRepr(*this); }

Array operator+(Scalar lhs, const Array& rhs) { return Add(lhs, rhs); }
Array operator-(Scalar lhs, const Array& rhs) { return Subtract(lhs, rhs); }
Array operator*(Scalar lhs, const Array& rhs) { return Multiply(lhs, rhs); }
Array operator/(Scalar lhs, const Array& rhs) { return Divide(lhs, rhs); }

namespace {

using internal::ArrayNode;
using internal::OpNode;

class PrintComputationalGraphImpl {
private:
    using VisitedArrayNodeSet = std::unordered_set<const ArrayNode*>;

    struct State {
        VisitedArrayNodeSet visited_array_nodes;
        int indent;
    };

    // TODO(niboshi): Make the options configurable from outside
    struct Options {
        bool print_metadata{true};
    };

public:
    explicit PrintComputationalGraphImpl(std::ostream& os) : os_{os} {}

    void Run(const ArrayNode& array_node, int indent) {
        State state{{}, indent};
        RunImpl(state, array_node);
    }

    std::string GetArrayNodeName(const ArrayNode& array_node) {
        static constexpr char kChars[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        static constexpr size_t kNumChars = sizeof(kChars) / sizeof(kChars[0]) - 1;
        static const auto kLen = static_cast<size_t>(std::ceil(sizeof(size_t) * 8U / std::log2(kNumChars)));
        auto it = array_name_map_.find(&array_node);
        if (it != array_name_map_.end()) {
            return it->second;
        }
        size_t hash = std::hash<const ArrayNode*>{}(&array_node);
        std::string s(kLen, '0');
        // Fill the string from left to right, because hash may be just the raw address and MSBs may be indistinguishable.
        for (auto it_s = s.begin(); hash > 0 && it_s != s.end(); ++it_s) {
            *it_s = gsl::at(kChars, hash % kNumChars);
            hash /= kNumChars;
        }
        return s;
    }

    std::string Indent(int indent) {
        static constexpr char kIndentChar = ' ';
        return std::string(static_cast<size_t>(indent * 2), kIndentChar);
    }

    void RunImpl(State& state, const ArrayNode& array_node) {
        std::string name = GetArrayNodeName(array_node);

        int indent = state.indent;
        VisitedArrayNodeSet& visited_array_nodes = state.visited_array_nodes;
        os_ << Indent(indent) << "ArrayNode<" << name << " " << &array_node << " rank=" << array_node.rank()
            << " shape=" << array_node.shape() << " dtype=" << GetDtypeName(array_node.dtype()) << ">" << std::endl;

        if (visited_array_nodes.end() == visited_array_nodes.find(&array_node)) {
            visited_array_nodes.insert(&array_node);

            if (options_.print_metadata) {
                std::shared_ptr<internal::ArrayBody> body = array_node.weak_body().lock();
                if (body == nullptr) {
                    os_ << Indent(indent + 2) << "body=(gone)" << std::endl;
                } else {
                    os_ << Indent(indent + 2) << "body=" << body.get() << std::endl;
                    const nonstd::optional<Array>* grad = body->GetGrad(array_node.backprop_id());
                    CHAINERX_ASSERT(grad != nullptr);
                    if (grad->has_value()) {
                        os_ << Indent(indent + 2) << "grad=<shape=" << (*grad)->shape() << " dtype=" << GetDtypeName((*grad)->dtype())
                            << ">" << std::endl;
                    }
                }
            }

            std::shared_ptr<const OpNode> op = array_node.creator_op_node();
            if (op) {
                os_ << Indent(indent + 1) << "Op<" << op->name() << " " << op.get() << " rank=" << op->rank() << ">" << std::endl;
                for (const std::shared_ptr<ArrayNode>& input_array_node : op->input_array_nodes()) {
                    state.indent += 2;
                    if (input_array_node != nullptr) {
                        RunImpl(state, *input_array_node);
                    } else {
                        os_ << Indent(state.indent) << "(null)" << std::endl;
                    }
                    state.indent -= 2;
                }
            }
        }
    }

    void SetArrayName(const ArrayNode& array_node, std::string name) { array_name_map_[&array_node] = std::move(name); }

private:
    std::ostream& os_;
    Options options_{};
    std::unordered_map<const ArrayNode*, std::string> array_name_map_;
};

}  // namespace

void DebugDumpComputationalGraph(
        std::ostream& os,
        const Array& array,
        const nonstd::optional<BackpropId>& backprop_id,
        int indent,
        const std::vector<std::pair<ConstArrayRef, std::string>>& array_name_map) {
    PrintComputationalGraphImpl impl{os};
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(array, backprop_id);
    for (const auto& pair : array_name_map) {
        for (const std::shared_ptr<ArrayNode>& array_node : internal::GetArrayBody(pair.first.get())->nodes()) {
            if (array_node->backprop_id() == actual_backprop_id) {
                impl.SetArrayName(*array_node, pair.second);
            }
        }
    }
    impl.Run(*internal::GetArrayBody(array)->GetArrayNode(actual_backprop_id), indent);
}

}  // namespace chainerx
