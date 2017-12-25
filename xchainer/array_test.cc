#include "xchainer/array.h"

#include <array>
#include <cstddef>
#include <initializer_list>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#include "xchainer/array_node.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/op_node.h"

namespace xchainer {
namespace {

class ArrayTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    virtual void SetUp() {
        std::string device_name = ::testing::get<0>(GetParam());
        device_scope_ = std::make_unique<DeviceScope>(device_name);
    }

    virtual void TearDown() { device_scope_.reset(); }

public:
    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::shared_ptr<void> data) {
        return {shape, TypeToDtype<T>, data};
    }

    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::initializer_list<T> data) {
        auto a = std::make_unique<T[]>(data.size());
        std::copy(data.begin(), data.end(), a.get());
        return {shape, TypeToDtype<T>, std::move(a)};
    }

    template <typename T>
    void AssertEqual(const Array& lhs, const Array& rhs) {
        ASSERT_NO_THROW(CheckEqual(lhs.dtype(), rhs.dtype()));
        ASSERT_NO_THROW(CheckEqual(lhs.shape(), rhs.shape()));
        auto total_size = lhs.shape().total_size();
        const T* ldata = static_cast<const T*>(lhs.data().get());
        const T* rdata = static_cast<const T*>(rhs.data().get());
        for (decltype(total_size) i = 0; i < total_size; i++) {
            ASSERT_EQ(ldata[i], rdata[i]);
        }
    }

    bool IsPointerCudaManaged(const void* ptr) {
#ifdef XCHAINER_ENABLE_CUDA
        cudaPointerAttributes attr = {};
        cuda::CheckError(cudaPointerGetAttributes(&attr, ptr));
        return attr.isManaged != 0;
#else
        (void)ptr;
        return false;
#endif  // XCHAINER_ENABLE_CUDA
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
};

TEST_P(ArrayTest, Ctor) {
    std::shared_ptr<void> data = std::make_unique<float[]>(2 * 3 * 4);
    Array x = MakeArray<float>({2, 3, 4}, data);
    ASSERT_EQ(TypeToDtype<float>, x.dtype());
    ASSERT_EQ(3, x.ndim());
    ASSERT_EQ(2 * 3 * 4, x.total_size());
    ASSERT_EQ(4, x.element_bytes());
    ASSERT_EQ(2 * 3 * 4 * 4, x.total_bytes());
    const std::shared_ptr<void> x_data = x.data();
    if (GetCurrentDevice() == MakeDevice("cpu")) {
        ASSERT_EQ(data, x_data);
    } else if (GetCurrentDevice() == MakeDevice("cuda")) {
        ASSERT_NE(data, x_data);
        ASSERT_TRUE(IsPointerCudaManaged(x_data.get()));
    } else {
        assert(0);
    }
    ASSERT_EQ(0, x.offset());
    ASSERT_TRUE(x.is_contiguous());
}

TEST_P(ArrayTest, ConstArray) {
    std::shared_ptr<void> data = std::make_unique<float[]>(2 * 3 * 4);
    const Array x = MakeArray<float>({2, 3, 4}, data);
    std::shared_ptr<const void> x_data = x.data();
    if (GetCurrentDevice() == MakeDevice("cpu")) {
        ASSERT_EQ(data, x_data);
    } else if (GetCurrentDevice() == MakeDevice("cuda")) {
        ASSERT_NE(data, x_data);
        ASSERT_TRUE(IsPointerCudaManaged(x_data.get()));
    } else {
        assert(0);
    }
}

TEST_P(ArrayTest, IAdd) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, true, false});
        a.IAdd(b);
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        a.IAdd(b);
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        a.IAdd(b);
        AssertEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, IMul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        a.IMul(b);
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        a.IMul(b);
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        a.IMul(b);
        AssertEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, Add) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, true, false});
        Array o = a.Add(b);
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        Array o = a.Add(b);
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        Array o = a.Add(b);
        AssertEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, Mul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        Array o = a.Mul(b);
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        Array o = a.Mul(b);
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        Array o = a.Mul(b);
        AssertEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ComputationalGraph) {
    {
        // c = a + b
        // o = a * c
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        {
            auto a_node = a.node();
            auto b_node = b.node();
            ASSERT_NE(a_node, nullptr) << "a's node is null";
            ASSERT_NE(b_node, nullptr) << "b's node is null";
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            ASSERT_EQ(a_op_node, nullptr) << "a's op node is not null";
            ASSERT_EQ(b_op_node, nullptr) << "b's op node is not null";
        }

        Array c = a.Add(b);
        {
            auto a_node = a.node();
            auto b_node = b.node();
            auto c_node = c.node();
            ASSERT_NE(a_node, nullptr) << "a's node is null";
            ASSERT_NE(b_node, nullptr) << "b's node is null";
            ASSERT_NE(c_node, nullptr) << "c's node is null";
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            auto c_op_node = c_node->next_node();
            ASSERT_EQ(a_op_node, nullptr) << "a's op node is not null";
            ASSERT_EQ(b_op_node, nullptr) << "b's op node is not null";
            ASSERT_NE(c_op_node, nullptr) << "c's op node is null";
            ASSERT_EQ(c_op_node->name(), "add") << "c's op node name is wrong";
        }

        Array o = a.Mul(c);
        {
            auto a_node = a.node();
            auto b_node = b.node();
            auto c_node = c.node();
            auto o_node = o.node();
            ASSERT_NE(a_node, nullptr) << "a's node is null";
            ASSERT_NE(b_node, nullptr) << "b's node is null";
            ASSERT_NE(c_node, nullptr) << "c's node is null";
            ASSERT_NE(o_node, nullptr) << "o's node is null";
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            auto c_op_node = c_node->next_node();
            auto o_op_node = o_node->next_node();
            ASSERT_EQ(a_op_node, nullptr) << "a's op node is not null";
            ASSERT_EQ(b_op_node, nullptr) << "b's op node is not null";
            ASSERT_NE(c_op_node, nullptr) << "c's op node is null";
            ASSERT_NE(o_op_node, nullptr) << "o's op node is null";
            ASSERT_EQ(c_op_node->name(), "add") << "c's op node name is wrong";
            ASSERT_EQ(o_op_node->name(), "mul") << "o's op node name is wrong";
        }
    }
}

TEST_P(ArrayTest, ComputationalGraphInplace) {
    {
        // a += b
        // a *= b
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        auto a_node_1 = a.node();
        {
            auto a_node = a_node_1;
            auto b_node = b.node();
            ASSERT_NE(a_node, nullptr) << "a's node is null";
            ASSERT_NE(b_node, nullptr) << "b's node is null";
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            ASSERT_EQ(a_op_node, nullptr) << "a's op node is not null";
            ASSERT_EQ(b_op_node, nullptr) << "b's op node is not null";
        }

        a.IAdd(b);
        auto a_node_2 = a.node();
        {
            auto a_node = a_node_2;
            auto b_node = b.node();
            ASSERT_NE(a_node, nullptr) << "a's node is null";
            ASSERT_NE(a_node, a_node_1) << "a's node is not changed";
            ASSERT_NE(b_node, nullptr) << "b's node is null";
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            ASSERT_NE(a_op_node, nullptr) << "a's op node is null";
            ASSERT_EQ(b_op_node, nullptr) << "b's op node is not null";
            ASSERT_EQ(a_op_node->name(), "add") << "a's op node name is wrong";
        }

        a.IMul(b);
        {
            auto a_node = a.node();
            auto b_node = b.node();
            ASSERT_NE(a_node, nullptr) << "a's node is null";
            ASSERT_NE(a_node, a_node_1) << "a's node is not changed";
            ASSERT_NE(a_node, a_node_2) << "a's node is not changed";
            ASSERT_NE(b_node, nullptr) << "b's node is null";
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            ASSERT_NE(a_op_node, nullptr) << "a's op node is null";
            ASSERT_EQ(b_op_node, nullptr) << "b's op node is not null";
            ASSERT_EQ(a_op_node->name(), "mul") << "a's op node name is wrong";
        }
    }
}

INSTANTIATE_TEST_CASE_P(ForEachDevice, ArrayTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                      std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                      std::string{"cpu"}));

}  // namespace
}  // namespace xchainer
