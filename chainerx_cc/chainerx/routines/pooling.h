#pragma once

#include <cstdint>
#include <memory>
#include <tuple>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/constant.h"
#include "chainerx/op.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

class MaxPoolGradState {
public:
    virtual ~MaxPoolGradState() = default;
};

class MaxPoolOp : public Op {
public:
    static const char* name() { return "MaxPool"; }

    virtual std::tuple<Array, std::unique_ptr<MaxPoolGradState>> Call(
            const Array& x,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            bool cover_all,
            bool return_state,
            const nonstd::optional<Array>& out) = 0;
};

class MaxPoolGradGradState {
public:
    virtual ~MaxPoolGradGradState() = default;
};

class MaxPoolGradOp : public Op {
public:
    static const char* name() { return "MaxPoolGrad"; }

    virtual std::tuple<Array, std::unique_ptr<MaxPoolGradGradState>> Call(
            const Array& gout,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            const std::shared_ptr<MaxPoolGradState>& state,
            bool return_state,
            const nonstd::optional<Array>& gx) = 0;
};

class MaxPoolGradGradOp : public Op {
public:
    static const char* name() { return "MaxPoolGradGrad"; }

    virtual Array Call(
            const Array& ggx,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            bool cover_all,
            const std::shared_ptr<MaxPoolGradGradState>& state,
            const nonstd::optional<Array>& ggout) = 0;
};

class AveragePoolGradState {
public:
    virtual ~AveragePoolGradState() = default;
};

class AveragePoolOp : public Op {
public:
    static const char* name() { return "AveragePool"; }

    virtual std::tuple<Array, std::unique_ptr<AveragePoolGradState>> Call(
            const Array& x,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            AveragePoolPadMode pad_mode,
            bool return_state,
            const nonstd::optional<Array>& out) = 0;
};

class AveragePoolGradOp : public Op {
public:
    static const char* name() { return "AveragePoolGrad"; }

    virtual Array Call(
            const Array& gout,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            AveragePoolPadMode pad_mode,
            const std::shared_ptr<AveragePoolGradState>& state,
            const nonstd::optional<Array>& gx) = 0;
};

Array MaxPool(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all = true);

enum class AveragePoolPadMode {
    kZero = 1,
    kIgnore,
};

Array AveragePool(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        AveragePoolPadMode pad_mode = AveragePoolPadMode::kIgnore);

}  // namespace chainerx
