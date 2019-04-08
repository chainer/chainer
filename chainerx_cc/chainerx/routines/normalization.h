#pragma once

#include <memory>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/constant.h"
#include "chainerx/op.h"
#include "chainerx/scalar.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

class BatchNormOp : public Op {
public:
    static const char* name() { return "BatchNormForward"; }

    // Intermediate state values such as the mean and inverse std can be written to `state` for reuse in BatchNormGradOp.
    virtual void Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& running_mean,
            const Array& running_var,
            Scalar eps,
            Scalar decay,
            const Axes& axis,
            const Array& out,
            nonstd::optional<std::shared_ptr<void>>& state) = 0;
};

class BatchNormGradOp : public Op {
public:
    static const char* name() { return "BatchNormBackward"; }

    virtual void Call(
            const Array& x,
            const Array& gamma,
            const Array& gout,
            Scalar eps,
            const Axes& axis,
            const Array& gx,
            const Array& ggamma,
            const Array& gbeta,
            nonstd::optional<std::shared_ptr<void>>& state) = 0;
};

class GenericBatchNormOp : public BatchNormOp {
public:
    void Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& running_mean,
            const Array& running_var,
            Scalar eps,
            Scalar decay,
            const Axes& axis,
            const Array& out,
            nonstd::optional<std::shared_ptr<void>>& state) override;
};

class GenericBatchNormGradOp : public BatchNormGradOp {
public:
    void Call(
            const Array& x,
            const Array& gamma,
            const Array& gout,
            Scalar eps,
            const Axes& axis,
            const Array& gx,
            const Array& ggamma,
            const Array& gbeta,
            nonstd::optional<std::shared_ptr<void>>& state) override;
};

class FixedBatchNormOp : public Op {
public:
    static const char* name() { return "FixedBatchNormForward"; }

    virtual void Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& mean,
            const Array& var,
            Scalar eps,
            const Axes& axis,
            const Array& out) = 0;
};

class GenericFixedBatchNormOp : public FixedBatchNormOp {
public:
    void Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& mean,
            const Array& var,
            Scalar eps,
            const Axes& axis,
            const Array& out) override;
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
