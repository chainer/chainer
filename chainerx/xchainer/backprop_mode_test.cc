#include "chainerx/backprop_mode.h"

#include <string>

#include <gtest/gtest.h>

#include "chainerx/backprop_scope.h"
#include "chainerx/constant.h"
#include "chainerx/context.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/context_session.h"
#include "chainerx/testing/device_session.h"

namespace chainerx {
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

}  // namespace
}  // namespace chainerx
