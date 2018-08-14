#include "xchainer/backprop_mode.h"

#include <string>

#include <gtest/gtest.h>

#include "xchainer/backprop_scope.h"
#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/context_session.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

TEST(BackpropModeScopeTest, NoBackpropModeScopeSingle) {
    testing::ContextSession context_session{};

    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    {
        NoBackpropModeScope scope{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_FALSE(IsBackpropRequired(backprop_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    {
        NoBackpropModeScope scope{backprop_id1, backprop_id2};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_FALSE(IsBackpropRequired(backprop_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    {
        NoBackpropModeScope scope{{}};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(IsBackpropRequired(backprop_id2));
}

TEST(BackpropModeScopeTest, ForceBackpropModeScopeSingle) {
    testing::ContextSession context_session{};

    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    {
        ForceBackpropModeScope scope{};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    {
        ForceBackpropModeScope scope{backprop_id1, backprop_id2};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    {
        ForceBackpropModeScope scope{{}};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(IsBackpropRequired(backprop_id2));
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultiple) {
    testing::ContextSession context_session{};

    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    {
        ForceBackpropModeScope scope1{backprop_id1};
        {
            NoBackpropModeScope scope2{backprop_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    {
        ForceBackpropModeScope scope1{backprop_id1};
        {
            ForceBackpropModeScope scope2{backprop_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{backprop_id1};
        {
            NoBackpropModeScope scope2{backprop_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{backprop_id1};
        {
            ForceBackpropModeScope scope2{backprop_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleVariedArgumentTypes) {
    testing::ContextSession context_session{};

    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_FALSE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_FALSE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_FALSE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_FALSE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{backprop_id1};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_FALSE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_FALSE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_FALSE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{backprop_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{backprop_id1};
        {
            NoBackpropModeScope scope2{};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_FALSE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{backprop_id1};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    {
        NoBackpropModeScope scope1{backprop_id1};
        {
            NoBackpropModeScope scope2{backprop_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleGraphArguments) {
    testing::ContextSession context_session{};

    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();

    {
        {
            NoBackpropModeScope scope1{backprop_id1, backprop_id2};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_FALSE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
    {
        {
            std::vector<BackpropId> backprop_ids{backprop_id1, backprop_id2};
            NoBackpropModeScope scope1{backprop_ids};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_FALSE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(IsBackpropRequired(backprop_id2));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeOneContext) {
    testing::ContextSession context_session{};

    BackpropScope backprop_scope1{"bp1"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();

    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(context_session.context()));
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        {
            ForceBackpropModeScope scope2{backprop_id1};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(context_session.context()));
            EXPECT_TRUE(IsBackpropRequired(backprop_id1));
            {
                NoBackpropModeScope scope3{backprop_id1};
                EXPECT_FALSE(IsBackpropRequired());
                EXPECT_FALSE(IsBackpropRequired(context_session.context()));
                EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            }
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(context_session.context()));
            EXPECT_TRUE(IsBackpropRequired(backprop_id1));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(context_session.context()));
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleContexts) {
    testing::ContextSession context_session1{};
    BackpropScope backprop_scope1{"bp1"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();

    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(context_session1.context()));
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
        {
            testing::ContextSession context_session2{};
            BackpropScope backprop_scope2{"bp2"};
            BackpropId backprop_id2 = backprop_scope2.backprop_id();

            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(context_session1.context()));
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(context_session2.context()));
            EXPECT_TRUE(IsBackpropRequired(backprop_id2));

            NoBackpropModeScope scope1{backprop_id2};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(context_session1.context()));
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(backprop_id2));
            {
                ForceBackpropModeScope scope2{backprop_id2};
                EXPECT_TRUE(IsBackpropRequired());
                EXPECT_FALSE(IsBackpropRequired(context_session1.context()));
                EXPECT_FALSE(IsBackpropRequired(backprop_id1));
                EXPECT_TRUE(IsBackpropRequired(context_session2.context()));
                EXPECT_TRUE(IsBackpropRequired(backprop_id2));
            }
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(context_session1.context()));
            EXPECT_FALSE(IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(IsBackpropRequired(context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(backprop_id2));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(context_session1.context()));
        EXPECT_FALSE(IsBackpropRequired(backprop_id1));
    }
}

// It is possible to use in flat scope because, in C++ spec, dtors are called in reverse order of ctors.
TEST(BackpropModeScopeTest, BackpropModeScopeFlatScope) {
    testing::ContextSession context_session{};

    BackpropScope backprop_scope1{"bp1"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();

    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(context_session.context()));

        ForceBackpropModeScope scope2{backprop_id1};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(context_session.context()));
        EXPECT_TRUE(IsBackpropRequired(backprop_id1));
    }
}

TEST(BackpropModeScopeTest, BackpropModeWithoutContext) {
    EXPECT_THROW({ NoBackpropModeScope{}; }, ContextError);
    EXPECT_THROW({ ForceBackpropModeScope{}; }, ContextError);
}

TEST(BackpropModeScopeTest, BackpropModeScopeWithAnotherContext) {
    testing::ContextSession context_session{};
    BackpropScope backprop_scope{"bp1"};
    BackpropId backprop_id = backprop_scope.backprop_id();

    Context another_context{};
    BackpropScope another_backprop_scope{"another_backprop", another_context};
    BackpropId another_backprop_id = another_backprop_scope.backprop_id();

    {
        EXPECT_TRUE(IsBackpropRequired(backprop_id));
        EXPECT_TRUE(IsBackpropRequired(another_backprop_id));
        {
            NoBackpropModeScope scope{another_context};
            EXPECT_TRUE(IsBackpropRequired(backprop_id));
            EXPECT_FALSE(IsBackpropRequired(another_backprop_id));
        }
    }
    {
        EXPECT_TRUE(IsBackpropRequired(backprop_id));
        EXPECT_TRUE(IsBackpropRequired(another_backprop_id));
        {
            NoBackpropModeScope scope{another_backprop_id};
            EXPECT_TRUE(IsBackpropRequired(backprop_id));
            EXPECT_FALSE(IsBackpropRequired(another_backprop_id));
        }
    }
    EXPECT_THROW(NoBackpropModeScope({backprop_id, another_backprop_id}), ContextError);
}

TEST(BackpropModeScopeTest, IsBackpropRequiredNoGraph) {
    testing::DeviceSession device_session{DeviceId{"native", 0}};
    Array a = testing::BuildArray({2, 1}).WithLinearData<float>();

    EXPECT_FALSE(a.IsBackpropRequired());
    EXPECT_FALSE(a.IsBackpropRequired(AnyGraph{}));
}

TEST(BackpropModeScopeTest, IsBackpropRequiredSingleGraph) {
    testing::DeviceSession device_session{DeviceId{"native", 0}};

    BackpropScope backprop_scope1{"bp1"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();

    Array a = testing::BuildArray({2, 1}).WithLinearData<float>();
    a.RequireGrad(backprop_id1);

    EXPECT_FALSE(a.IsBackpropRequired());
    EXPECT_TRUE(a.IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(a.IsBackpropRequired(AnyGraph{}));
    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(a.IsBackpropRequired());
        EXPECT_FALSE(a.IsBackpropRequired(backprop_id1));
        EXPECT_FALSE(a.IsBackpropRequired(AnyGraph{}));
        {
            ForceBackpropModeScope scope2{backprop_id1};
            EXPECT_FALSE(a.IsBackpropRequired());
            EXPECT_TRUE(a.IsBackpropRequired(backprop_id1));
            EXPECT_TRUE(a.IsBackpropRequired(AnyGraph{}));
        }
    }
}

TEST(BackpropModeScopeTest, IsBackpropRequiredMultipleGraphs) {
    testing::DeviceSession device_session{DeviceId{"native", 0}};

    BackpropScope backprop_scope1{"bp1"};
    BackpropScope backprop_scope2{"bp2"};
    BackpropScope backprop_scope3{"bp3"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();
    BackpropId backprop_id2 = backprop_scope2.backprop_id();
    BackpropId backprop_id3 = backprop_scope3.backprop_id();

    Array a = testing::BuildArray({2, 1}).WithLinearData<float>();
    a.RequireGrad(backprop_id1);
    a.RequireGrad(backprop_id2);

    EXPECT_TRUE(a.IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(a.IsBackpropRequired(backprop_id2));
    EXPECT_TRUE(a.IsBackpropRequired(AnyGraph{}));
    {
        NoBackpropModeScope scope1{backprop_id1};
        EXPECT_FALSE(a.IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(a.IsBackpropRequired(backprop_id2));
        EXPECT_TRUE(a.IsBackpropRequired(AnyGraph{}));
        {
            NoBackpropModeScope scope2{backprop_id2};
            EXPECT_FALSE(a.IsBackpropRequired(backprop_id1));
            EXPECT_FALSE(a.IsBackpropRequired(backprop_id2));
            EXPECT_FALSE(a.IsBackpropRequired(AnyGraph{}));
            {
                ForceBackpropModeScope scope3{backprop_id1};
                EXPECT_TRUE(a.IsBackpropRequired(backprop_id1));
                EXPECT_FALSE(a.IsBackpropRequired(backprop_id2));
                EXPECT_TRUE(a.IsBackpropRequired(AnyGraph{}));
            }
            {
                ForceBackpropModeScope scope3{backprop_id2};
                EXPECT_FALSE(a.IsBackpropRequired(backprop_id1));
                EXPECT_TRUE(a.IsBackpropRequired(backprop_id2));
                EXPECT_TRUE(a.IsBackpropRequired(AnyGraph{}));
            }
            {
                ForceBackpropModeScope scope3{{backprop_id3}};
                EXPECT_FALSE(a.IsBackpropRequired(backprop_id1));
                EXPECT_FALSE(a.IsBackpropRequired(backprop_id2));
                EXPECT_FALSE(a.IsBackpropRequired(AnyGraph{}));
            }
        }
    }
    {
        NoBackpropModeScope scope{};
        EXPECT_FALSE(a.IsBackpropRequired(backprop_id1));
        EXPECT_FALSE(a.IsBackpropRequired(backprop_id2));
        EXPECT_FALSE(a.IsBackpropRequired(AnyGraph{}));
    }
    {
        NoBackpropModeScope scope{backprop_id1, backprop_id2};
        EXPECT_FALSE(a.IsBackpropRequired(backprop_id1));
        EXPECT_FALSE(a.IsBackpropRequired(backprop_id2));
        EXPECT_FALSE(a.IsBackpropRequired(AnyGraph{}));
    }
}

TEST(BackpropModeScopeTest, IsBackpropRequiredAnotherContext) {
    testing::DeviceSession device_session{DeviceId{"native", 0}};

    BackpropScope backprop_scope1{"bp1"};
    BackpropId backprop_id1 = backprop_scope1.backprop_id();

    Array a = testing::BuildArray({2, 1}).WithLinearData<float>();
    a.RequireGrad(backprop_id1);

    EXPECT_FALSE(a.IsBackpropRequired());
    EXPECT_TRUE(a.IsBackpropRequired(backprop_id1));
    EXPECT_TRUE(a.IsBackpropRequired(AnyGraph{}));
    {
        testing::ContextSession another_context_session{};
        NoBackpropModeScope scope{};
        // BackpropModeScope of another context does not reflect.
        EXPECT_FALSE(a.IsBackpropRequired());
        EXPECT_TRUE(a.IsBackpropRequired(backprop_id1));
        EXPECT_TRUE(a.IsBackpropRequired(AnyGraph{}));
    }
}

}  // namespace
}  // namespace xchainer
