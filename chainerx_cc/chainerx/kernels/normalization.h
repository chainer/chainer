#pragma once

#include <memory>
#include <tuple>
#include <utility>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/dtype.h"
#include "chainerx/kernel.h"
#include "chainerx/scalar.h"

namespace chainerx {

// Intermediate results from `BatchNormKernel::Call` can be stored in this construct and be reused in `BatchNormGradKernel::Call`.
// The objects to store may vary depending on backend so each backend should derive this class to define the actual set of intermediate
// results.
class BatchNormGradState {
public:
    BatchNormGradState() = default;

    virtual ~BatchNormGradState() = default;

    BatchNormGradState(const BatchNormGradState&) = delete;
    BatchNormGradState(BatchNormGradState&&) = delete;
    BatchNormGradState& operator=(const BatchNormGradState&) = delete;
    BatchNormGradState& operator=(BatchNormGradState&&) = delete;
};

class BatchNormKernel : public Kernel {
public:
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
            const absl::optional<Array>& out) = 0;
};

class BatchNormGradKernel : public Kernel {
public:
    // Returns gx, ggamma, gbeta.
    virtual std::tuple<Array, Array, Array> Call(
            const Array& x,
            const Array& gamma,
            const Array& gout,
            Scalar eps,
            const Axes& axis,
            const std::shared_ptr<BatchNormGradState>& state,
            const absl::optional<Array>& gx,
            const absl::optional<Array>& ggamma,
            const absl::optional<Array>& gbeta) = 0;
};

class GenericBatchNormGradState : public BatchNormGradState {
public:
    GenericBatchNormGradState(Array x_mean, Array x_inv_std, Dtype beta_dtype)
        : x_mean_{std::move(x_mean)}, x_inv_std_{std::move(x_inv_std)}, beta_dtype_{beta_dtype} {}

    const Array& x_mean() const { return x_mean_; }
    const Array& x_inv_std() const { return x_inv_std_; }
    Dtype beta_dtype() const { return beta_dtype_; }

private:
    Array x_mean_;
    Array x_inv_std_;
    Dtype beta_dtype_;
};

class GenericBatchNormKernel : public BatchNormKernel {
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
            const absl::optional<Array>& out) override;
};

class GenericBatchNormGradKernel : public BatchNormGradKernel {
public:
    std::tuple<Array, Array, Array> Call(
            const Array& x,
            const Array& gamma,
            const Array& gout,
            Scalar eps,
            const Axes& axis,
            const std::shared_ptr<BatchNormGradState>& state,
            const absl::optional<Array>& gx,
            const absl::optional<Array>& ggamma,
            const absl::optional<Array>& gbeta) override;
};

class FixedBatchNormKernel : public Kernel {
public:
    virtual Array Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& mean,
            const Array& var,
            Scalar eps,
            const Axes& axis,
            const absl::optional<Array>& out) = 0;
};

class GenericFixedBatchNormKernel : public FixedBatchNormKernel {
public:
    Array Call(
            const Array& x,
            const Array& gamma,
            const Array& beta,
            const Array& mean,
            const Array& var,
            Scalar eps,
            const Axes& axis,
            const absl::optional<Array>& out) override;
};

}  // namespace chainerx
