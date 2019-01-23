#include "chainerx/crossplatform.h"

#include <string>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "chainerx/error.h"

namespace chainerx {
namespace {

class CrossplatformEnvTest : public ::testing::Test {
protected:
    void SetUp() override { orig_value_ = crossplatform::GetEnv(name_); }

    void TearDown() override {
        if (orig_value_.has_value()) {
            crossplatform::SetEnv(name_, *orig_value_);
        }
    }

    const std::string& name() { return name_; }

private:
    std::string name_{"DUMMY_CHAINERX_PATH_"};
    nonstd::optional<std::string> orig_value_{};
};

TEST_F(CrossplatformEnvTest, Env) {
    std::string value{"/home/chainerx"};

    crossplatform::SetEnv(name(), value);
    nonstd::optional<std::string> retrieved_value = crossplatform::GetEnv(name());

    EXPECT_TRUE(retrieved_value.has_value());
    EXPECT_EQ(value, *retrieved_value);

    crossplatform::UnsetEnv(name());
    retrieved_value = crossplatform::GetEnv(name());
    EXPECT_FALSE(retrieved_value.has_value());
}

TEST(CrossplatformInvalidEnvTest, EnvInvalidName) { EXPECT_THROW(crossplatform::SetEnv("invalid=name", "value"), ChainerxError); }

}  // namespace
}  // namespace chainerx
