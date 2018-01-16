#include "check_backward.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#include "xchainer/error.h"
#include "xchainer/gradient_detail.h"
#include "xchainer/numeric.h"

namespace xchainer {

// TODO(hvy): mock dependent functions
namespace test {

void Backprop(const Arrays& inputs, const Arrays& grad_outputs) {
    if (inputs.size() != grad_outputs.size()) {
        throw std::runtime_error("mismatch");
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
      // TODO(hvy): set gradients
    }
}

Arrays CalculateNumericalGradient(ForwardFunction func, const Arrays& inputs, const Arrays& grad_outputs, const Arrays& eps) {
    Arrays grads;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(grads), [](const Array& x) { return x; });
    return grads;
}

}  // namepace test

template <typename T, typename U>
void Zip(const T& lhs, const T& rhs, U op) {
    for (size_t i = 0; i < lhs.size(); ++i) {
        op(lhs[i], rhs[i]);
    }
}

void CheckBackwardComputation(const ForwardFunction& func, const std::vector<Array>& inputs, const std::vector<Array>& grad_outputs,
                              const std::vector<Array>& eps, float atol, float rtol) {
    std::vector<Array> outputs = func(inputs);

    // TODO(hvy): delete debug print
    std::cout << "output size: " << outputs.size() << std::endl;
    for (auto o : outputs) {
        std::cout << "output: " << o.data() << " " << o.shape() << std::endl;
    }

    // TODO(hvy): call some backward function
    // test::Backprop(outputs, grad_outputs);
    test::Backprop(inputs, grad_outputs);

    // TODO(hvy): keep a copy/reference to the computed input gradients
    std::vector<Array> grads;
    // std::transform(inputs.begin(), inputs.end(), std::back_inserter(grads), [](const Array& input) { return *input.node()->grad(); });

    // TODO(hvy): call numerical_grad with given function and eps (specified per element) to get the numerical gradient
    std::vector<Array> numerical_grads = test::CalculateNumericalGradient(func, inputs, grad_outputs, eps);

    for (size_t i = 0; i < grads.size(); ++i) {
        if (!AllClose(grads[i], numerical_grads[i], atol, rtol)) {
            // TODO(hvy): write proper message
            throw AssertionError("too large errors");
        }
    }
}

}  // namespace xchainer
