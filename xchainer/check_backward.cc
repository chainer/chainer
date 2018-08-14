#include "xchainer/check_backward.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_body_leak_detection.h"
#include "xchainer/array_node.h"
#include "xchainer/backprop_mode.h"
#include "xchainer/backward.h"
#include "xchainer/backward_builder.h"
#include "xchainer/error.h"
#include "xchainer/numeric.h"
#include "xchainer/numerical_gradient.h"

namespace xchainer {
namespace {

// Disconnects the graphs of input arrays.
// RequireGrad() is configured to match the original arrays, but not connected to them.
// This prevents the graph nodes generated within check functions from leaking to the original input arrays, which would interfere with the
// leak detector.
std::vector<Array> DisconnectInputArrays(const std::vector<Array>& inputs) {
    std::vector<Array> inputs_view;
    inputs_view.reserve(inputs.size());
    for (const auto& input : inputs) {
        Array a = input.AsGradStopped(CopyKind::kView);
        for (const std::shared_ptr<internal::ArrayNode>& arr_node : internal::GetArrayBody(input)->nodes()) {
            a.RequireGrad(arr_node->backprop_id());
        }
        inputs_view.emplace_back(std::move(a));
    }
    return inputs_view;
}

std::vector<nonstd::optional<Array>> BackwardGradients(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        std::vector<Array>& inputs,
        const nonstd::optional<std::vector<Array>>& grad_outputs,
        const BackpropId& backprop_id,
        DoubleBackpropOption double_backprop = DoubleBackpropOption::kEnable) {
    for (const auto& input : inputs) {
        const std::shared_ptr<internal::ArrayBody>& input_body = internal::GetArrayBody(input);
        if (input_body->HasArrayNode(backprop_id) && input_body->GetArrayNode(backprop_id)->creator_op_node() != nullptr) {
            throw GradientCheckError{"BackwardGradients: All inputs must be leaf nodes of computational graph"};
        }
    }

    std::vector<Array> outputs = func(inputs);

    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < outputs.size(); ++j) {
            if (internal::GetArrayBody(inputs[i]) == internal::GetArrayBody(outputs[j]) && inputs[i].IsBackpropRequired(backprop_id)) {
                throw GradientCheckError{"BackwardGradients: Input ", i, " and output ", j, " of the forward function are identical."};
            }
        }
    }

    if (grad_outputs.has_value()) {
        const std::size_t nout = outputs.size();
        if (nout != grad_outputs->size()) {
            throw GradientCheckError{"BackwardGradients: Size of function outputs: ",
                                     nout,
                                     " and size of grad outputs: ",
                                     grad_outputs->size(),
                                     " must be same"};
        }

        for (std::size_t i = 0; i < nout; ++i) {
            if (outputs[i].IsBackpropRequired(backprop_id)) {
                outputs[i].SetGrad((*grad_outputs)[i], backprop_id);
            }
        }
    }

    // Clear gradients which may exist if func calls backward inside of itself.
    for (Array& input : inputs) {
        if (input.IsBackpropRequired(backprop_id)) {
            input.ClearGrad(backprop_id);
        }
    }

    std::vector<ConstArrayRef> output_refs;
    std::transform(outputs.begin(), outputs.end(), std::back_inserter(output_refs), [](const Array& output) {
        return std::reference_wrapper<const Array>{output};
    });
    Backward(output_refs, backprop_id, double_backprop);

    std::vector<nonstd::optional<Array>> backward_grads;
    std::transform(
            inputs.begin(),
            inputs.end(),
            std::back_inserter(backward_grads),
            [&backprop_id](const Array& input) -> nonstd::optional<Array> {
                if (!input.IsBackpropRequired(backprop_id)) {
                    return nonstd::nullopt;
                }
                return input.GetGrad(backprop_id);
            });

    return backward_grads;
}

void CheckDoubleBackpropOption(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const BackpropId& backprop_id) {
    std::ostringstream failure_os;

    // make it nonlinear to be double differentiable so that this utility can be used even for non double differentiable functions
    auto nonlinear_func = [&func](const std::vector<Array>& func_inputs) {
        std::vector<Array> nonlinear_outputs;
        for (const auto& output : func(func_inputs)) {
            nonlinear_outputs.emplace_back(output * output);
        }
        return nonlinear_outputs;
    };

    // Disable double backprop
    {
        std::vector<Array> inputs_disconnected = DisconnectInputArrays(inputs);
        std::vector<nonstd::optional<Array>> grads =
                BackwardGradients(nonlinear_func, inputs_disconnected, nonstd::nullopt, backprop_id, DoubleBackpropOption::kDisable);

        for (size_t i = 0; i < grads.size(); ++i) {
            if (grads[i]) {
                if (grads[i]->IsBackpropRequired(backprop_id)) {
                    failure_os << "Gradient " << i << " / " << grads.size() << " is connected to the graph '" << backprop_id
                               << "' even when double-backprop is disabled.";
                }
            }
        }
    }

    // Enable double backprop
    {
        std::vector<Array> inputs_disconnected = DisconnectInputArrays(inputs);
        std::vector<nonstd::optional<Array>> grads =
                BackwardGradients(nonlinear_func, inputs_disconnected, nonstd::nullopt, backprop_id, DoubleBackpropOption::kEnable);

        for (size_t i = 0; i < grads.size(); ++i) {
            if (grads[i]) {
                if (!grads[i]->IsBackpropRequired(backprop_id)) {
                    failure_os << "Gradient " << i << " / " << grads.size() << " is not connected to the graph '" << backprop_id
                               << "' even when double-backprop is enabled.";
                }
            }
        }
    }

    // Do nothing unless failure
    std::string failure_message = failure_os.str();
    if (!failure_message.empty()) {
        throw GradientCheckError{failure_message};
    }
}

void CheckBackwardComputation(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& grad_outputs,
        const std::vector<Array>& eps,
        double atol,
        double rtol,
        const nonstd::optional<BackpropId>& backprop_id) {
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(inputs.front(), backprop_id);

    // Compute backward gradients
    std::vector<Array> inputs_disconnected = DisconnectInputArrays(inputs);
    const std::vector<nonstd::optional<Array>> backward_grads =
            BackwardGradients(func, inputs_disconnected, grad_outputs, actual_backprop_id, DoubleBackpropOption::kDisable);
    if (backward_grads.size() != inputs.size()) {
        throw GradientCheckError{"Number of input gradients does not match the input arrays."};
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!backward_grads[i].has_value()) {
            continue;
        }
        const Array& backward_grad = *backward_grads[i];
        if (backward_grad.shape() != inputs[i].shape()) {
            throw GradientCheckError{"Shape of input gradient ",
                                     i,
                                     " of ",
                                     inputs.size(),
                                     " ",
                                     backward_grad.shape(),
                                     " does not match the corresponding input shape ",
                                     inputs[i].shape(),
                                     "."};
        }
        if (backward_grad.dtype() != inputs[i].dtype()) {
            throw GradientCheckError{"Dtype of input gradient ",
                                     i,
                                     " of ",
                                     inputs.size(),
                                     " ",
                                     GetDtypeName(backward_grad.dtype()),
                                     " does not match the corresponding input dtype ",
                                     GetDtypeName(inputs[i].dtype()),
                                     "."};
        }
    }

    // Compute numerical gradients
    const std::vector<Array> numerical_grads = CalculateNumericalGradient(func, inputs, grad_outputs, eps);

    // If you're trapped in any of these asserts, numerical gradiends must be implemented incorrectly.
    assert(numerical_grads.size() == inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        assert(numerical_grads[i].shape() == inputs[i].shape());
        assert(numerical_grads[i].dtype() == inputs[i].dtype());
    }

    // Check numerical consistency between numerical and backward gradients.
    std::vector<size_t> failed_input_indices;
    for (size_t i = 0; i < backward_grads.size(); ++i) {
        if (!backward_grads[i].has_value()) {
            continue;
        }
        if (!AllClose(*backward_grads[i], numerical_grads[i], atol, rtol)) {
            failed_input_indices.emplace_back(i);
        }
    }
    if (!failed_input_indices.empty()) {
        std::ostringstream os;
        os << "Numerical error in backward on inputs (out of " << inputs.size() << "): ";
        for (size_t i : failed_input_indices) {
            if (i != 0) {
                os << ", ";
            }
            os << i;
        }
        os << std::endl;
        os << "Backprop ID: " << actual_backprop_id << std::endl;
        os << "Atol: " << atol << "  Rtol: " << rtol << std::endl;
        for (size_t i : failed_input_indices) {
            os << "Error[" << i << "]:" << std::endl
               << *backward_grads[i] - numerical_grads[i] << std::endl  // TODO(niboshi): Use abs
               << "Backward gradients[" << i << "]:" << std::endl
               << *backward_grads[i] << std::endl
               << "Numerical gradients[" << i << "]:" << std ::endl
               << numerical_grads[i] << std::endl
               << "Eps[" << i << "] (perturbation in numerical gradients):" << std::endl
               << eps[i] << std::endl;
        }
        throw GradientCheckError{os.str()};
    }
}

}  // namespace

namespace {

// Asserts all the array bodies are freed in the leak tracker.
void CheckAllArrayBodiesFreed(internal::ArrayBodyLeakTracker& tracker) {
    std::ostringstream os;
    if (!tracker.IsAllArrayBodiesFreed(os)) {
        throw GradientCheckError{os.str()};
    }
}

}  // namespace

void CheckBackward(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& grad_outputs,
        const std::vector<Array>& eps,
        double atol,
        double rtol,
        const nonstd::optional<BackpropId>& backprop_id) {
#ifndef NDEBUG
    assert(!inputs.empty());
    assert(std::all_of(inputs.begin(), inputs.end(), [&backprop_id](const Array& a) { return a.IsBackpropRequired(backprop_id); }));

    assert(!grad_outputs.empty());
    assert(std::none_of(
            grad_outputs.begin(), grad_outputs.end(), [&backprop_id](const Array& a) { return a.IsBackpropRequired(backprop_id); }));

    assert(eps.size() == inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        assert(eps[i].shape() == inputs[i].shape());
        assert(&eps[i].device() == &inputs[i].device());
    }
#endif

    BackpropId actual_backprop_id = internal::GetArrayBackpropId(inputs.front(), backprop_id);

    {
        internal::ArrayBodyLeakTracker tracker{};
        {
            internal::ArrayBodyLeakDetectionScope scope{tracker};
            CheckDoubleBackpropOption(func, inputs, actual_backprop_id);
        }
        CheckAllArrayBodiesFreed(tracker);
    }

    {
        internal::ArrayBodyLeakTracker tracker{};
        {
            internal::ArrayBodyLeakDetectionScope scope{tracker};
            CheckBackwardComputation(func, inputs, grad_outputs, eps, atol, rtol, backprop_id);
        }
        CheckAllArrayBodiesFreed(tracker);
    }
}

namespace {

void CheckDoubleBackwardComputationImpl(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& grad_outputs,
        const std::vector<Array>& grad_grad_inputs,
        const std::vector<Array>& eps,
        double atol,
        double rtol,
        const nonstd::optional<BackpropId>& backprop_id) {
    BackpropId actual_backprop_id = internal::GetArrayBackpropId(inputs.front(), backprop_id);
    const std::size_t nin = inputs.size();
    const std::size_t nout = grad_outputs.size();

    if (grad_grad_inputs.size() != nin) {
        throw XchainerError{"Number of input arrays and grad_grad_input arrays do not match."};
    }

    // LIMITATION: All inputs must require gradients unlike CheckBackwardComputation

    // Check all the input arrays require gradients
    for (size_t i = 0; i < nin; ++i) {
        if (!inputs[i].IsBackpropRequired(actual_backprop_id)) {
            throw XchainerError{"Input array ", i, " / ", nin, " is not differentiable w.r.t. the backprop ID '", actual_backprop_id, "'."};
        }
    }

    // Check all the output gradient arrays require gradients
    for (size_t i = 0; i < nout; ++i) {
        if (!grad_outputs[i].IsBackpropRequired(actual_backprop_id)) {
            throw XchainerError{
                    "Output gradient array ", i, " / ", nout, " is not differentiable w.r.t. the backprop ID '", actual_backprop_id, "'."};
        }
    }

    // The "forward" function to return the first order gradients

    auto first_order_grad_func = [&func, nin, nout, &actual_backprop_id](const std::vector<Array>& inputs_and_grad_outputs) {
        // Just revert (split) inputs_and_grad_outputs into inputs and grad_outputs
        std::vector<Array> inputs{inputs_and_grad_outputs.begin(), inputs_and_grad_outputs.begin() + nin};
        std::vector<Array> grad_outputs{inputs_and_grad_outputs.begin() + nin, inputs_and_grad_outputs.end()};

        ForceBackpropModeScope scope{actual_backprop_id};

        for (Array& input : inputs) {
            input.RequireGrad(actual_backprop_id);
        }

        // Compute first order gradients
        std::vector<nonstd::optional<Array>> optional_backward_grads = BackwardGradients(func, inputs, grad_outputs, actual_backprop_id);

        // Check all the first order gradients are computed
        if (optional_backward_grads.size() != nin) {
            throw GradientCheckError{"Number of first-order input gradients arrays ",
                                     optional_backward_grads.size(),
                                     " do not match the number of input arrays ",
                                     nin,
                                     "."};
        }

        for (size_t i = 0; i < nin; ++i) {
            if (!optional_backward_grads[i].has_value()) {
                throw GradientCheckError{"First-order input gradient ", i, " / ", nin, " does not exist."};
            }
        }

        for (size_t i = 0; i < nin; ++i) {
            if (!optional_backward_grads[i]->IsBackpropRequired(actual_backprop_id)) {
                throw GradientCheckError{"First-order Input gradient ",
                                         i,
                                         " / ",
                                         nin,
                                         " is not differentiable w.r.t. the backprop ID '",
                                         actual_backprop_id,
                                         "'."};
            }
        }

        // Convert to std::vector<Array>
        std::vector<Array> backward_grads;
        std::transform(
                optional_backward_grads.begin(),
                optional_backward_grads.end(),
                std::back_inserter(backward_grads),
                [](const nonstd::optional<Array>& optional_backward_grad) { return *optional_backward_grad; });

        assert(backward_grads.size() == nin);
        return backward_grads;
    };

    // Prepare for computing numerical and backward gradients.
    // Merge inputs and grad_outputs into inputs_and_grad_outputs.
    std::vector<Array> inputs_and_grad_outputs;
    inputs_and_grad_outputs.reserve(nin + nout);
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputs_and_grad_outputs));
    std::copy(grad_outputs.begin(), grad_outputs.end(), std::back_inserter(inputs_and_grad_outputs));

    CheckBackwardComputation(first_order_grad_func, inputs_and_grad_outputs, grad_grad_inputs, eps, atol, rtol, backprop_id);
}

}  // namespace

void CheckDoubleBackwardComputation(
        const std::function<std::vector<Array>(const std::vector<Array>&)>& func,
        const std::vector<Array>& inputs,
        const std::vector<Array>& grad_outputs,
        const std::vector<Array>& grad_grad_inputs,
        const std::vector<Array>& eps,
        double atol,
        double rtol,
        const nonstd::optional<BackpropId>& backprop_id) {
#ifndef NDEBUG
    assert(!inputs.empty());
    assert(std::all_of(inputs.begin(), inputs.end(), [&backprop_id](const Array& a) { return a.IsBackpropRequired(backprop_id); }));

    assert(!grad_outputs.empty());
    assert(std::all_of(
            grad_outputs.begin(), grad_outputs.end(), [&backprop_id](const Array& a) { return a.IsBackpropRequired(backprop_id); }));

    assert(grad_grad_inputs.size() == inputs.size());
    assert(std::none_of(grad_grad_inputs.begin(), grad_grad_inputs.end(), [&backprop_id](const Array& a) {
        return a.IsBackpropRequired(backprop_id);
    }));
    for (size_t i = 0; i < inputs.size(); ++i) {
        assert(inputs[i].shape() == grad_grad_inputs[i].shape());
        assert(inputs[i].dtype() == grad_grad_inputs[i].dtype());
        assert(&inputs[i].device() == &grad_grad_inputs[i].device());
    }

    assert(eps.size() == inputs.size() + grad_outputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        assert(eps[i].shape() == inputs[i].shape());
        assert(&eps[i].device() == &inputs[i].device());
    }
    for (size_t i = 0; i < grad_outputs.size(); ++i) {
        assert(eps[inputs.size() + i].shape() == grad_outputs[i].shape());
        assert(&eps[inputs.size() + i].device() == &grad_outputs[i].device());
    }
#endif

    internal::ArrayBodyLeakTracker tracker{};
    {
        internal::ArrayBodyLeakDetectionScope scope{tracker};
        CheckDoubleBackwardComputationImpl(
                func, DisconnectInputArrays(inputs), DisconnectInputArrays(grad_outputs), grad_grad_inputs, eps, atol, rtol, backprop_id);
    }
    CheckAllArrayBodiesFreed(tracker);
}

}  // namespace xchainer
