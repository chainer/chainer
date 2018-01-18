#include "xchainer/gradient_check.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <gsl/gsl>

#include "xchainer/array.h"
#include "xchainer/array_repr.h"
#include "xchainer/check_backward.h"
#include "xchainer/device.h"
#include "xchainer/op_node.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace {

Arrays CorrectUnaryFunc(const Arrays& inputs) {
    const Array& lhs = inputs[0];

    Array out = Array::EmptyLike(lhs);
    out.set_requires_grad(lhs.requires_grad());

    if (out.requires_grad()) {
        std::shared_ptr<ArrayNode> lhs_node = lhs.mutable_node();
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        int64_t out_rank = lhs_node->rank();
        auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{lhs_node};
        std::function<Array(const Array&)> empty_func;
        auto lhs_func = lhs.requires_grad() ? [lhs](const Array& gout) { return gout * lhs; } : empty_func;
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func};
        std::shared_ptr<OpNode> op_node = std::make_shared<OpNode>("correct_unary", out_rank, next_nodes, backward_functions);
        out_node->set_next_node(op_node);
        out_node->set_rank(out_rank + 1);
    }

    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        int64_t total_size = lhs.total_size();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());
        auto half = static_cast<T>(0.5);

        for (int64_t i = 0; i < total_size; i++) {
            odata[i] = ldata[i] * ldata[i] * half;
        }
    });

    return {out};
}

Arrays BrokenUnaryFunc(const Arrays& inputs) {
    const Array& lhs = inputs[0];

    Array out = Array::EmptyLike(lhs);
    out.set_requires_grad(lhs.requires_grad());

    if (out.requires_grad()) {
        std::shared_ptr<ArrayNode> lhs_node = lhs.mutable_node();
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        int64_t out_rank = lhs_node->rank();
        auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{lhs_node};
        std::function<Array(const Array&)> empty_func;
        auto lhs_func = lhs.requires_grad() ? [](const Array& gout) { return gout * gout; } : empty_func;
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func};
        std::shared_ptr<OpNode> op_node = std::make_shared<OpNode>("broken_unary", out_rank, next_nodes, backward_functions);
        out_node->set_next_node(op_node);
        out_node->set_rank(out_rank + 1);
    }

    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        int64_t total_size = lhs.total_size();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());

        for (int64_t i = 0; i < total_size; i++) {
            odata[i] = ldata[i];
        }
    });

    return {out};
}

Arrays CorrectBinaryFunc(const Arrays& inputs) {
    const Array& lhs = inputs[0];
    const Array& rhs = inputs[1];

    CheckEqual(lhs.dtype() == rhs.dtype());
    CheckEqual(lhs.shape() == rhs.shape());

    Array out = Array::EmptyLike(lhs);
    out.set_requires_grad(lhs.requires_grad() || rhs.requires_grad());

    if (out.requires_grad()) {
        std::shared_ptr<ArrayNode> lhs_node = lhs.mutable_node();
        std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node();
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        int64_t out_rank = std::max(lhs_node->rank(), rhs_node->rank());
        auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{lhs_node, rhs_node};
        std::function<Array(const Array&)> empty_func;
        auto lhs_func = lhs.requires_grad() ? [rhs](const Array& gout) { return gout * rhs; } : empty_func;
        auto rhs_func = rhs.requires_grad() ? [lhs](const Array& gout) { return gout * lhs; } : empty_func;
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
        std::shared_ptr<OpNode> op_node = std::make_shared<OpNode>("correct_binary", out_rank, next_nodes, backward_functions);
        out_node->set_next_node(op_node);
        out_node->set_rank(out_rank + 1);
    }

    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        int64_t total_size = lhs.total_size();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());

        for (int64_t i = 0; i < total_size; i++) {
            odata[i] = ldata[i] * rdata[i];
        }
    });

    return {out};
}

Arrays BrokenBinaryFunc(const Arrays& inputs) {
    const Array& lhs = inputs[0];
    const Array& rhs = inputs[1];

    CheckEqual(lhs.dtype() == rhs.dtype());
    CheckEqual(lhs.shape() == rhs.shape());

    Array out = Array::EmptyLike(lhs);
    out.set_requires_grad(lhs.requires_grad() || rhs.requires_grad());

    if (out.requires_grad()) {
        std::shared_ptr<ArrayNode> lhs_node = lhs.mutable_node();
        std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node();
        std::shared_ptr<ArrayNode> out_node = out.RenewNode();
        int64_t out_rank = std::max(lhs_node->rank(), rhs_node->rank());
        auto next_nodes = std::vector<std::shared_ptr<ArrayNode>>{lhs_node, rhs_node};
        std::function<Array(const Array&)> empty_func;
        auto lhs_func = lhs.requires_grad() ? [rhs](const Array& gout) { return gout + rhs; } : empty_func;
        auto rhs_func = rhs.requires_grad() ? [lhs](const Array& gout) { return gout + lhs; } : empty_func;
        auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
        std::shared_ptr<OpNode> op_node = std::make_shared<OpNode>("test_mul", out_rank, next_nodes, backward_functions);
        out_node->set_next_node(op_node);
        out_node->set_rank(out_rank + 1);
    }

    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        int64_t total_size = lhs.total_size();
        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());

        for (int64_t i = 0; i < total_size; i++) {
            odata[i] = ldata[i] * rdata[i];
        }
    });

    return {out};
}

class CheckBackwardTest : public ::testing::Test {
public:
    using Arrays = std::vector<Array>;

    template <typename T>
    Array MakeArray(const Shape& shape, const T* data) {
        int64_t size = shape.total_size();
        auto a = std::make_unique<T[]>(size);
        std::copy(data, data + size, a.get());
        return Array::FromBuffer(shape, TypeToDtype<T>, std::move(a));
    }

    void CheckCorrectBackwardComputation(const Arrays& inputs, const Arrays& grad_outputs, const Arrays& eps, double atol, double rtol) {
        auto fprop = inputs.size() == 1 ? &CorrectUnaryFunc : &CorrectBinaryFunc;
        CheckBackwardComputation(fprop, inputs, grad_outputs, eps, atol, rtol);
    }

    void CheckBrokenBackwardComputation(const Arrays& inputs, const Arrays& grad_outputs, const Arrays& eps, double atol, double rtol) {
        auto fprop = inputs.size() == 1 ? &BrokenUnaryFunc : &BrokenBinaryFunc;
        if (std::any_of(inputs.begin(), inputs.end(), [](const Array& input) { return input.requires_grad(); })) {
            // Catch the gtest failure expected to be generated by CheckBackwardComputation but without failing this test
            EXPECT_NONFATAL_FAILURE(CheckBackwardComputation(fprop, inputs, grad_outputs, eps, atol, rtol), "Backward check failure");
        } else {
            // We cannot expect any failures in case none of the input Arrays require gradients
            CheckBackwardComputation(fprop, inputs, grad_outputs, eps, atol, rtol);
        }
    }
};

TEST_F(CheckBackwardTest, UnaryBackward) {
    Shape shape{2, 3};

    float data[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float eps_data[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float grad_output_data[]{1.f, -2.f, 3.f, 0.f, 5.2f, 6.f};

    double atol = 1e-5;
    double rtol = 1e-4;

    {
        Arrays inputs = {MakeArray(shape, data)};
        Arrays eps = {MakeArray(shape, eps_data)};
        Arrays grad_outputs = {MakeArray(shape, grad_output_data)};
        inputs[0].set_requires_grad(false);
        CheckCorrectBackwardComputation(inputs, grad_outputs, eps, atol, rtol);
    }
    {
        Arrays inputs = {MakeArray(shape, data)};
        Arrays eps = {MakeArray(shape, eps_data)};
        Arrays grad_outputs = {MakeArray(shape, grad_output_data)};
        inputs[0].set_requires_grad(false);
        CheckBrokenBackwardComputation(inputs, grad_outputs, eps, atol, rtol);
    }
    {
        Arrays inputs = {MakeArray(shape, data)};
        Arrays eps = {MakeArray(shape, eps_data)};
        Arrays grad_outputs = {MakeArray(shape, grad_output_data)};
        inputs[0].set_requires_grad(true);
        CheckCorrectBackwardComputation(inputs, grad_outputs, eps, atol, rtol);
    }
    {
        Arrays inputs = {MakeArray(shape, data)};
        Arrays eps = {MakeArray(shape, eps_data)};
        Arrays grad_outputs = {MakeArray(shape, grad_output_data)};
        inputs[0].set_requires_grad(true);
        CheckBrokenBackwardComputation(inputs, grad_outputs, eps, atol, rtol);
    }
}

TEST_F(CheckBackwardTest, BinaryBackward) {
    Shape shape{2, 3};

    float data1[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float data2[]{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    float eps1[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float eps2[]{3.f, -2.f, 3.f, -4.f, 3.2f, 0.9f};
    float grad_output_data[]{1.f, -2.f, 3.f, 0.f, 5.2f, 6.f};

    double atol = 1e-5;
    double rtol = 1e-4;

    const int nperm = 4;
    bool requires_grads[nperm][2]{{false, false}, {false, true}, {true, false}, {true, true}};

    for (int i = 0; i < nperm; ++i) {
        {
            Arrays inputs = {MakeArray(shape, data1), MakeArray(shape, data2)};
            Arrays eps = {MakeArray(shape, eps1), MakeArray(shape, eps2)};
            Arrays grad_outputs = {MakeArray(shape, grad_output_data)};

            inputs[0].set_requires_grad(requires_grads[i][0]);
            inputs[1].set_requires_grad(requires_grads[i][1]);

            CheckCorrectBackwardComputation(inputs, grad_outputs, eps, atol, rtol);
        }
        {
            Arrays inputs = {MakeArray(shape, data1), MakeArray(shape, data2)};
            Arrays eps = {MakeArray(shape, eps1), MakeArray(shape, eps2)};
            Arrays grad_outputs = {MakeArray(shape, grad_output_data)};

            inputs[0].set_requires_grad(requires_grads[i][0]);
            inputs[1].set_requires_grad(requires_grads[i][1]);

            CheckBrokenBackwardComputation(inputs, grad_outputs, eps, atol, rtol);
        }
    }
}

}  // namespace
}  // namespace xchainer
