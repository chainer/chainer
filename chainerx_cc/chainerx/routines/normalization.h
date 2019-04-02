#pragma once

#include <array>
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
    class ForwardBackward {
    public:
        virtual ~ForwardBackward() = default;
        virtual Array Forward(const Array& x, const Array& gamma, const Array& beta) = 0;
        virtual std::array<Array, 3> Backward(const Array& gout) = 0;
    };

    static const char* name() { return "BatchNorm"; }

    virtual Array Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& running_mean,
            const Array& running_var,
            Scalar eps,
            Scalar decay,
            const OptionalAxes& axis);

protected:
    virtual std::unique_ptr<ForwardBackward> GetForwardBackward(
            const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, const Axes& axis) = 0;
};

class FixedBatchNormOp : public Op {
public:
    static const char* name() { return "FixedBatchNorm"; }

    virtual Array
    Call(const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const OptionalAxes& axis);

protected:
    virtual Array Impl(
            const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis) = 0;
};

class GenericBatchNormOp : public BatchNormOp {
public:
    class GenericForwardBackward : public ForwardBackward {
    public:
        GenericForwardBackward(const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, Axes axis);

        Array Forward(const Array& x, const Array& gamma, const Array& beta) override;
        std::array<Array, 3> Backward(const Array& gout) override;

    protected:
        void SetForwardResults(Array x, Array gamma, Array x_mean, Array x_inv_std, Dtype beta_dtype);

        const Array& running_mean() { return running_mean_; }
        const Array& running_var() { return running_var_; }
        Scalar eps() { return eps_; }
        Scalar decay() { return decay_; }
        const Axes& axis() { return axis_; }

        // Forward results.
        const Array& x() { return *x_; }
        const Array& gamma() { return *gamma_; }
        const Array& x_mean() { return *x_mean_; }
        const Array& x_inv_std() { return *x_inv_std_; }
        Dtype beta_dtype() {
            if (!beta_dtype_.has_value()) {
                throw ChainerxError{"Beta dtype must first be set with a call to SetForwardResults."};
            }
            return *beta_dtype_;
        }

    private:
        const Array& running_mean_;
        const Array& running_var_;
        Scalar eps_;
        Scalar decay_;
        Axes axis_;

        // TODO(niboshi): Fix header dependency order and hold arrays directly.
        std::shared_ptr<Array> x_;
        std::shared_ptr<Array> gamma_;
        std::shared_ptr<Array> x_mean_;
        std::shared_ptr<Array> x_inv_std_;
        nonstd::optional<Dtype> beta_dtype_;
    };

protected:
    std::unique_ptr<ForwardBackward> GetForwardBackward(
            const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, const Axes& axis) override;
};

class GenericFixedBatchNormOp : public FixedBatchNormOp {
protected:
    Array Impl(const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis)
            override;
};

// Computes the batch normalization along the given axis.
// If axis is omitted, the first axis is treated as the batch axis and will be reduced during normalization.
// Running mean and running variance that are passed as arguments will be updated in-place.
inline Array BatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps = 2e-5,
        Scalar decay = 0.9,
        const OptionalAxes& axis = nonstd::nullopt) {
    return x.device().backend().CallOp<BatchNormOp>(x, gamma, beta, running_mean, running_var, eps, decay, axis);
}

// Computes the fixed batch normalization.
// axis argument is treated in the same way as BatchNorm.
// Backward computation is not implemented.
inline Array FixedBatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& mean,
        const Array& var,
        Scalar eps,
        const OptionalAxes& axis = nonstd::nullopt) {
    return x.device().backend().CallOp<FixedBatchNormOp>(x, gamma, beta, mean, var, eps, axis);
}

}  // namespace chainerx
