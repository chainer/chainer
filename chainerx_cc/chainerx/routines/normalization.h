#pragma once

#include <memory>
#include <tuple>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/constant.h"
#include "chainerx/op.h"
#include "chainerx/scalar.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

// Intermediate results from `BatchNormOp::Call` can be stored in this construct and be reused in `BatchNormGradOp::Call`.
// The objects to store may vary depending on backend so each backend should derive this class to define the actual set of intermediate
// results.
class BatchNormGradState {
public:
    virtual ~BatchNormGradState() = default;
};

class BatchNormOp : public Op {
public:
    static const char* name() { return "BatchNorm"; }

    // The returned state should be a `nullptr` if `return_state` is `false`.
    virtual std::tuple<Array, std::unique_ptr<BatchNormGradState>> Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& running_mean,
            const Array& running_var,
            Scalar eps,
            Scalar decay,
            const Axes& axis,
            bool return_state,
            const nonstd::optional<Array>& out) = 0;
};

class BatchNormGradOp : public Op {
public:
    static const char* name() { return "BatchNormGrad"; }

    // Returns gx, ggamma, gbeta.
    virtual std::tuple<Array, Array, Array> Call(
            const Array& x,
            const Array& gamma,
            const Array& gout,
            Scalar eps,
            const Axes& axis,
            const std::shared_ptr<BatchNormGradState>& state,
            const nonstd::optional<Array>& gx,
            const nonstd::optional<Array>& ggamma,
            const nonstd::optional<Array>& gbeta) = 0;
};

class GenericBatchNormOp : public BatchNormOp {
public:
    std::tuple<Array, std::unique_ptr<BatchNormGradState>> Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& running_mean,
            const Array& running_var,
            Scalar eps,
            Scalar decay,
            const Axes& axis,
            bool return_state,
            const nonstd::optional<Array>& out) override;
};

class GenericBatchNormGradOp : public BatchNormGradOp {
public:
    std::tuple<Array, Array, Array> Call(
            const Array& x,
            const Array& gamma,
            const Array& gout,
            Scalar eps,
            const Axes& axis,
            const std::shared_ptr<BatchNormGradState>& state,
            const nonstd::optional<Array>& gx,
            const nonstd::optional<Array>& ggamma,
            const nonstd::optional<Array>& gbeta) override;
};

class FixedBatchNormOp : public Op {
public:
    static const char* name() { return "FixedBatchNorm"; }

    virtual Array Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& mean,
            const Array& var,
            Scalar eps,
            const Axes& axis,
            const nonstd::optional<Array>& out) = 0;
};

class GenericFixedBatchNormOp : public FixedBatchNormOp {
public:
    Array Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& mean,
            const Array& var,
            Scalar eps,
            const Axes& axis,
            const nonstd::optional<Array>& out) override;
};

// Computes the batch normalization along the given axis.
// If axis is omitted, the first axis is treated as the batch axis and will be reduced during normalization.
// Running mean and running variance that are passed as arguments will be updated in-place.
Array BatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps = 2e-5,
        Scalar decay = 0.9,
        const OptionalAxes& axis = nonstd::nullopt);

// Computes the fixed batch normalization.
// axis argument is treated in the same way as BatchNorm.
// Backward computation is not implemented.
Array FixedBatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& mean,
        const Array& var,
        Scalar eps,
        const OptionalAxes& axis = nonstd::nullopt);

}  // namespace chainerx
