#include "xchainer/check_backward.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>
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
                                                       const std::vector<Array>& inputs, const std::vector<Array>& grad_outputs) {
    std::vector<Array> outputs = func(inputs);

    const int nout = outputs.size();
    for (int i = 0; i < nout; ++i) {
        outputs[i].mutable_node()->set_grad(grad_outputs[i]);
    }

    // TODO(hvy): Currently only supporting functions with single outputs, support any number of outputs instead
    if (outputs.size() > 1) {
        throw NotImplementedError("Functions with more than one output are not supported");
    }
    Backward(outputs[0]);

    std::vector<nonstd::optional<Array>> backward_grads;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(backward_grads), [](const Array& input) { return input.grad(); });

    return backward_grads;
}

}  // namespace

void CheckBackwardComputation(std::function<std::vector<Array>(const std::vector<Array>&)> func, const std::vector<Array>& inputs,
                              const std::vector<Array>& grad_outputs, const std::vector<Array>& eps, double atol, double rtol) {
    const std::vector<Array> numerical_grads = CalculateNumericalGradient(func, inputs, grad_outputs, eps);
    const std::vector<nonstd::optional<Array>> backward_grads = BackwardGradients(func, inputs, grad_outputs);
    ASSERT_EQ(backward_grads.size(), numerical_grads.size());

    std::ostringstream failure_os;
    const int nin = backward_grads.size();
    for (int i = 0; i < nin; ++i) {
        if (backward_grads[i]) {  // All inputs do not necessarily require gradients
            if (!AllClose(*backward_grads[i], numerical_grads[i], atol, rtol)) {
                failure_os << "Backward check failure on input " << i << " (Total inputs: " << inputs.size() << ")\n"
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

}  // namespace xchainer
