#include "xchainer/check_backward.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/backprop.h"
#include "xchainer/error.h"
#include "xchainer/gradient_check.h"
#include "xchainer/numeric.h"

namespace xchainer {
namespace {

std::vector<nonstd::optional<Array>> BackwardGradients(std::function<std::vector<Array>(const std::vector<Array>&)> func,
                                                       const std::vector<Array>& inputs, const std::vector<Array>& grad_outputs,
                                                       const GraphId& graph_id) {
    for (auto& input : inputs) {
        if (internal::HasArrayNode(input, graph_id) && internal::GetArrayNode(input, graph_id)->next_node()) {
            throw XchainerError("BackwardGradients: All inputs must be leaf nodes of computational graph");
        }
    }

    std::vector<Array> outputs = func(inputs);

    const std::size_t nout = outputs.size();
    if (nout != grad_outputs.size()) {
        throw XchainerError("BackwardGradients: Size of function outputs: " + std::to_string(nout) + " and size of grad outputs: " +
                            std::to_string(grad_outputs.size()) + " must be same");
    }

    for (std::size_t i = 0; i < nout; ++i) {
        if (outputs[i].IsGradRequired(graph_id)) {
            outputs[i].SetGrad(grad_outputs[i], graph_id);
        }
    }

    // Clear gradients which may exist if func calls backward inside of itself.
    for (Array& input : const_cast<std::vector<Array>&>(inputs)) {
        if (input.IsGradRequired(graph_id)) {
            input.ClearGrad(graph_id);
        }
    }

    // TODO(sonots): Use new Backward API which accepts a vector of arrays
    for (auto& output : outputs) {
        if (output.IsGradRequired(graph_id)) {
            Backward(output, graph_id, DoubleBackpropOption::kEnable);
        }
    }

    std::vector<nonstd::optional<Array>> backward_grads;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(backward_grads),
                   [&graph_id](const Array& input) -> nonstd::optional<Array> {
                       if (input.IsGradRequired(graph_id)) {
                           return input.GetGrad(graph_id);
                       } else {
                           return nonstd::nullopt;
                       }
                   });

    return backward_grads;
}

}  // namespace

void CheckBackwardComputation(std::function<std::vector<Array>(const std::vector<Array>&)> func, const std::vector<Array>& inputs,
                              const std::vector<Array>& grad_outputs, const std::vector<Array>& eps, double atol, double rtol,
                              const GraphId& graph_id) {
    const std::vector<Array> numerical_grads = CalculateNumericalGradient(func, inputs, grad_outputs, eps, graph_id);
    const std::vector<nonstd::optional<Array>> backward_grads = BackwardGradients(func, inputs, grad_outputs, graph_id);
    ASSERT_EQ(backward_grads.size(), numerical_grads.size());

    std::ostringstream failure_os;
    const int nin = backward_grads.size();
    for (int i = 0; i < nin; ++i) {
        if (backward_grads[i]) {  // All inputs do not necessarily require gradients
            if (!AllClose(*backward_grads[i], numerical_grads[i], atol, rtol)) {
                failure_os << "Backward check failure on input " << i << " (Total inputs: " << inputs.size() << ")\n"
                           << "Graph name: " << graph_id << "\n"
                           << "Atol: " << atol << "\n"
                           << "Rtol: " << rtol << "\n"
                           << "Eps (perturbation):\n"
                           << eps[i] << "\n"
                           << "Backward gradients:\n"
                           << *backward_grads[i] << "\n"
                           << "Numerical gradients:\n"
                           << numerical_grads[i];
            }
        }
    }

    // Do nothing if all backward-numerical gradient pairs were close, else generate a nonfatal failure
    std::string failure_message = failure_os.str();
    if (!failure_message.empty()) {
        ADD_FAILURE() << failure_message;
    }
}

// Test twice differentiation of a given procedure.
//
// This function automatically checks if the backward procedure of `func` is
// correctly implemented for further differentiation. It first computes the
// gradient of `func` w.r.t. its inputs in the same way as `CheckBackward`.
// This function then further invokes the backward procedure against the
// gradient variables, starting from the initial gradient given by `grad_grad_inputs`.
// It also computes the second gradient using `CalculateNumericalGradient`.
// The resulting gradients are compared to confirm if the second-order gradients
// are approximately correct.
//
// Note that this function **DOES NOT** check if the first-order differentiation
// is correct; the numerical gradient assumes that the first-order gradient given
// by the usual `Backward` is correct. The implementation of each differentiable
// function should be tested by `CheckBackward` first, and then should be tested
// by this function if neccessary.
void CheckDoubleBackwardComputation(std::function<std::vector<Array>(const std::vector<Array>&)> func, const std::vector<Array>& inputs,
                                    const std::vector<Array>& grad_outputs, const std::vector<Array>& grad_grad_inputs,
                                    const std::vector<Array>& eps, double atol, double rtol, const GraphId& graph_id) {
    // LIMITATION: All inputs must require gradients unlike CheckBackwardComputation
    std::for_each(inputs.begin(), inputs.end(), [&graph_id](auto& input) {
        if (!input.IsGradRequired(graph_id)) {
            throw XchainerError("All inputs must require gradients");
        }
    });

    const std::size_t nin = inputs.size();
    const std::size_t n_grad_outputs = grad_outputs.size();

    // Just merge inputs and grad_outputs into inputs_and_grad_outputs to pass into below `first_order_grad_func`.
    // Since move assignment operator of Array is deleted, we can not use std::vector::insert. Instead use reserve, and std::copy
    std::vector<Array> inputs_and_grad_outputs;
    inputs_and_grad_outputs.reserve(nin + n_grad_outputs);
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputs_and_grad_outputs));
    std::copy(grad_outputs.begin(), grad_outputs.end(), std::back_inserter(inputs_and_grad_outputs));

    auto first_order_grad_func = [func, nin, n_grad_outputs, graph_id](const std::vector<Array>& inputs_and_grad_outputs) {
        // Just revert (split) inputs_and_grad_outputs into inputs and grad_outputs
        std::vector<Array> inputs(inputs_and_grad_outputs.begin(), inputs_and_grad_outputs.begin() + nin);
        std::vector<Array> grad_outputs(inputs_and_grad_outputs.begin() + nin, inputs_and_grad_outputs.end());

        std::vector<nonstd::optional<Array>> optional_backward_grads = BackwardGradients(func, inputs, grad_outputs, graph_id);

        // Just convert std::vector<nonstd::optional<Array>> to std::vector<Array> so that CalculateNumericalGradient can accept
        std::vector<Array> backward_grads;
        std::transform(optional_backward_grads.begin(), optional_backward_grads.end(), std::back_inserter(backward_grads),
                       [&graph_id](const nonstd::optional<Array>& optional_backward_grad) {
                           if (optional_backward_grad.has_value()) {
                               return *optional_backward_grad;
                           } else {
                               throw XchainerError("All gradients must exist");
                           }
                       });
        return backward_grads;
    };

    const std::vector<Array> numerical_grads =
        CalculateNumericalGradient(first_order_grad_func, inputs_and_grad_outputs, grad_grad_inputs, eps, graph_id);
    const std::vector<nonstd::optional<Array>> backward_grads =
        BackwardGradients(first_order_grad_func, inputs_and_grad_outputs, grad_grad_inputs, graph_id);
    ASSERT_EQ(backward_grads.size(), numerical_grads.size());

    std::ostringstream failure_os;
    const int n_backward_grads = backward_grads.size();
    for (int i = 0; i < n_backward_grads; ++i) {
        if (!AllClose(*backward_grads[i], numerical_grads[i], atol, rtol)) {
            failure_os << "Backward check failure on input " << i << " (Total inputs: " << inputs.size() << ")\n"
                       << "Graph name: " << graph_id << "\n"
                       << "Atol: " << atol << "\n"
                       << "Rtol: " << rtol << "\n"
                       << "Eps (perturbation):\n"
                       << eps[i] << "\n"
                       << "Backward gradients:\n"
                       << *backward_grads[i] << "\n"
                       << "Numerical gradients:\n"
                       << numerical_grads[i];
        }
    }

    // Do nothing if all backward-numerical gradient pairs were close, else generate a nonfatal failure
    std::string failure_message = failure_os.str();
    if (!failure_message.empty()) {
        ADD_FAILURE() << failure_message;
    }
}

}  // namespace xchainer
