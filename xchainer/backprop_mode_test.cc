#include "xchainer/backprop_mode.h"

#include <string>

#include <gtest/gtest.h>

#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/testing/context_session.h"

namespace xchainer {
namespace {

TEST(BackpropModeScopeTest, NoBackpropModeScopeSingle) {
    testing::ContextSession context_session{};

    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired("graph1"));
    {
        NoBackpropModeScope scope{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired("graph1"));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired("graph1"));
    EXPECT_TRUE(IsBackpropRequired("graph2"));
    {
        NoBackpropModeScope scope{{"graph1", "graph2"}};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired("graph1"));
        EXPECT_FALSE(IsBackpropRequired("graph2"));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired("graph1"));
    EXPECT_TRUE(IsBackpropRequired("graph2"));
    {
        NoBackpropModeScope scope{{}};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired("graph1"));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired("graph1"));
}

TEST(BackpropModeScopeTest, ForceBackpropModeScopeSingle) {
    testing::ContextSession context_session{};

    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired("graph1"));
    {
        ForceBackpropModeScope scope{};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired("graph1"));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired("graph1"));
    EXPECT_TRUE(IsBackpropRequired("graph2"));
    {
        ForceBackpropModeScope scope{{"graph1", "graph2"}};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired("graph1"));
        EXPECT_TRUE(IsBackpropRequired("graph2"));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired("graph1"));
    EXPECT_TRUE(IsBackpropRequired("graph2"));
    {
        ForceBackpropModeScope scope{{}};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired("graph1"));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired("graph1"));
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultiple) {
    testing::ContextSession context_session{};

    {
        ForceBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired("graph1"));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired("graph1"));
    }
    {
        ForceBackpropModeScope scope1{{"graph1"}};
        {
            ForceBackpropModeScope scope2{{"graph1"}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired("graph1"));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired("graph1"));
    }
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired("graph1"));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired("graph1"));
    }
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            ForceBackpropModeScope scope2{{"graph1"}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired("graph1"));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired("graph1"));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleVariedArgumentTypes) {
    testing::ContextSession context_session{};

    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{};
            EXPECT_FALSE(IsBackpropRequired());
        }
        EXPECT_FALSE(IsBackpropRequired());
    }
    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_FALSE(IsBackpropRequired());
        }
        EXPECT_FALSE(IsBackpropRequired());
    }
    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired("graph"));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired("graph"));
    }
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{};
            EXPECT_FALSE(IsBackpropRequired());
        }
        EXPECT_TRUE(IsBackpropRequired());
    }
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_TRUE(IsBackpropRequired());
        }
        EXPECT_TRUE(IsBackpropRequired());
    }
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired("graph1"));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired("graph1"));
    }
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired("graph1"));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired("graph1"));
    }
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired("graph1"));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired("graph1"));
    }
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired("graph1"));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired("graph1"));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeOneContext) {
    testing::ContextSession context_session{};

    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session.context()));
        {
            ForceBackpropModeScope scope2{{"graph1"}};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session.context()));
            EXPECT_TRUE(IsBackpropRequired("graph1"));
            EXPECT_TRUE(IsBackpropRequired("graph1", context_session.context()));
            {
                NoBackpropModeScope scope3{{"graph1"}};
                EXPECT_FALSE(IsBackpropRequired());
                EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session.context()));
                EXPECT_FALSE(IsBackpropRequired("graph1"));
                EXPECT_FALSE(IsBackpropRequired("graph1", context_session.context()));
            }
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session.context()));
            EXPECT_TRUE(IsBackpropRequired("graph1"));
            EXPECT_TRUE(IsBackpropRequired("graph1", context_session.context()));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session.context()));
        EXPECT_FALSE(IsBackpropRequired("graph1"));
        EXPECT_FALSE(IsBackpropRequired("graph1", context_session.context()));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleContexts) {
    testing::ContextSession context_session1{};

    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session1.context()));
        {
            // New context stack, and a stack for the context should be created.
            testing::ContextSession context_session2{};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(kDefaultGraphId, context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session1.context()));

            NoBackpropModeScope scope1{{"graph1"}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(kDefaultGraphId, context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired("graph1"));
            EXPECT_FALSE(IsBackpropRequired("graph1", context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session1.context()));
            EXPECT_FALSE(IsBackpropRequired("graph1", context_session1.context()));
            {
                ForceBackpropModeScope scope2{{"graph1"}};
                EXPECT_TRUE(IsBackpropRequired());
                EXPECT_TRUE(IsBackpropRequired(kDefaultGraphId, context_session2.context()));
                EXPECT_TRUE(IsBackpropRequired("graph1"));
                EXPECT_TRUE(IsBackpropRequired("graph1", context_session2.context()));
                EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session1.context()));
                EXPECT_FALSE(IsBackpropRequired("graph1", context_session1.context()));
            }
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(kDefaultGraphId, context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired("graph1"));
            EXPECT_FALSE(IsBackpropRequired("graph1", context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session1.context()));
            EXPECT_FALSE(IsBackpropRequired("graph1", context_session1.context()));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session1.context()));
    }
}

// It is possible to use in flat scope because, in C++ spec, dtors are called in reverse order of ctors.
TEST(BackpropModeScopeTest, BackpropModeScopeFlatScope) {
    testing::ContextSession context_session{};

    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session.context()));

        ForceBackpropModeScope scope2{{"graph1"}};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(kDefaultGraphId, context_session.context()));
        EXPECT_TRUE(IsBackpropRequired("graph1"));
        EXPECT_TRUE(IsBackpropRequired("graph1", context_session.context()));
    }
}

TEST(BackpropModeScopeTest, BackpropModeWithoutContext) {
    EXPECT_THROW({ NoBackpropModeScope{}; }, ContextError);
    EXPECT_THROW({ ForceBackpropModeScope{}; }, ContextError);
}

}  // namespace
}  // namespace xchainer
