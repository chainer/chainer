#include "xchainer/gradient_check.h"

#include <memory>
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

    template <typename T>
    void CheckCheckBackwardComputation(bool expect_allclose, T forward) {
        Shape shape{2, 3};
        float data1[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        float data2[]{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        float eps1[]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        float eps2[]{3.f, -2.f, 3.f, -4.f, 3.2f, 0.9f};
        float grad_output_data[]{1.f, -2.f, 3.f, 0.f, 5.2f, 6.f};
        float atol = 1e-5;
        float rtol = 1e-4;

        Arrays inputs = {
            MakeArray(shape, data1), MakeArray(shape, data2),
        };
        Arrays eps = {
            MakeArray(shape, eps1), MakeArray(shape, eps2),
        };
        Arrays grad_outputs = {
            MakeArray(shape, grad_output_data),
        };

        inputs[0].set_requires_grad(true);
        inputs[1].set_requires_grad(true);

        if (expect_allclose) {
            CheckBackwardComputation(forward, inputs, grad_outputs, eps, atol, rtol);
        } else {
            EXPECT_NONFATAL_FAILURE(CheckBackwardComputation(forward, inputs, grad_outputs, eps, atol, rtol), "Backward check failure");
        }
    }
};

TEST_F(CheckBackwardTest, CorrectBackward) {
    auto forward = [](const Arrays& inputs) -> Arrays {
        const Array& lhs = inputs[0];
        const Array& rhs = inputs[1];

        Array out = Array::EmptyLike(lhs);
        out.set_requires_grad(lhs.requires_grad() || rhs.requires_grad());

        if (out.requires_grad()) {
            std::shared_ptr<ArrayNode> lhs_node = lhs.mutable_node();
            std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node();
            std::shared_ptr<ArrayNode> out_node = out.RenewNode();

            std::function<Array(const Array&)> empty_func;
            auto lhs_func = lhs.requires_grad() ? [rhs](const Array& gout) { return gout * rhs; } : empty_func;
            auto rhs_func = rhs.requires_grad() ? [lhs](const Array& gout) { return gout * lhs; } : empty_func;
            auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
            std::shared_ptr<OpNode> op_node =
                std::make_shared<OpNode>("test_mul", 0, std::vector<std::shared_ptr<ArrayNode>>{lhs_node, rhs_node}, backward_functions);
            out_node->set_next_node(op_node);
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
    };

    CheckCheckBackwardComputation(true, forward);
}

TEST_F(CheckBackwardTest, IncorrectBackward) {
    auto forward = [](const Arrays& inputs) -> Arrays {
        const Array& lhs = inputs[0];
        const Array& rhs = inputs[1];

        Array out = Array::EmptyLike(lhs);
        out.set_requires_grad(lhs.requires_grad() || rhs.requires_grad());

        if (out.requires_grad()) {
            std::shared_ptr<ArrayNode> lhs_node = lhs.mutable_node();
            std::shared_ptr<ArrayNode> rhs_node = rhs.mutable_node();
            std::shared_ptr<ArrayNode> out_node = out.RenewNode();

            std::function<Array(const Array&)> empty_func;
            auto lhs_func = lhs.requires_grad() ? [rhs](const Array& gout) { return gout * rhs; } : empty_func;
            auto rhs_func = rhs.requires_grad() ? [lhs](const Array& gout) { return gout * lhs; } : empty_func;
            auto backward_functions = std::vector<std::function<Array(const Array&)>>{lhs_func, rhs_func};
            std::shared_ptr<OpNode> op_node =
                std::make_shared<OpNode>("test_mul", 0, std::vector<std::shared_ptr<ArrayNode>>{lhs_node, rhs_node}, backward_functions);
            out_node->set_next_node(op_node);
        }

        VisitDtype(lhs.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;

            int64_t total_size = lhs.total_size();
            auto* ldata = static_cast<const T*>(lhs.data().get());
            auto* rdata = static_cast<const T*>(rhs.data().get());
            auto* odata = static_cast<T*>(out.data().get());

            for (int64_t i = 0; i < total_size; i++) {
                odata[i] = ldata[i] + rdata[i];
            }
        });

        return {out};
    };

    CheckCheckBackwardComputation(false, forward);
}

}  // namespace
}  // namespace xchainer
