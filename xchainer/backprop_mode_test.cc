#include "xchainer/backprop_mode.h"

#include <string>

#include <gtest/gtest.h>

#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/graph_scope.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/context_session.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

TEST(BackpropModeScopeTest, NoBackpropModeScopeSingle) {
    testing::ContextSession context_session{};

    GraphScope graph_scope1{"graph1"};
    GraphScope graph_scope2{"graph2"};
    GraphId graph_id1 = graph_scope1.graph_id();
    GraphId graph_id2 = graph_scope2.graph_id();

    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(graph_id1));
    EXPECT_TRUE(IsBackpropRequired(graph_id2));
    {
        NoBackpropModeScope scope{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_FALSE(IsBackpropRequired(graph_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(graph_id1));
    EXPECT_TRUE(IsBackpropRequired(graph_id2));
    {
        NoBackpropModeScope scope{graph_id1, graph_id2};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_FALSE(IsBackpropRequired(graph_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(graph_id1));
    EXPECT_TRUE(IsBackpropRequired(graph_id2));
    {
        NoBackpropModeScope scope{{}};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(graph_id1));
    EXPECT_TRUE(IsBackpropRequired(graph_id2));
}

TEST(BackpropModeScopeTest, ForceBackpropModeScopeSingle) {
    testing::ContextSession context_session{};

    GraphScope graph_scope1{"graph1"};
    GraphScope graph_scope2{"graph2"};
    GraphId graph_id1 = graph_scope1.graph_id();
    GraphId graph_id2 = graph_scope2.graph_id();

    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(graph_id1));
    EXPECT_TRUE(IsBackpropRequired(graph_id2));
    {
        ForceBackpropModeScope scope{};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(graph_id1));
    EXPECT_TRUE(IsBackpropRequired(graph_id2));
    {
        ForceBackpropModeScope scope{graph_id1, graph_id2};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(graph_id1));
    EXPECT_TRUE(IsBackpropRequired(graph_id2));
    {
        ForceBackpropModeScope scope{{}};
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    EXPECT_TRUE(IsBackpropRequired());
    EXPECT_TRUE(IsBackpropRequired(graph_id1));
    EXPECT_TRUE(IsBackpropRequired(graph_id2));
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultiple) {
    testing::ContextSession context_session{};

    GraphScope graph_scope1{"graph1"};
    GraphScope graph_scope2{"graph2"};
    GraphId graph_id1 = graph_scope1.graph_id();
    GraphId graph_id2 = graph_scope2.graph_id();

    {
        ForceBackpropModeScope scope1{graph_id1};
        {
            NoBackpropModeScope scope2{graph_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    {
        ForceBackpropModeScope scope1{graph_id1};
        {
            ForceBackpropModeScope scope2{graph_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{graph_id1};
        {
            NoBackpropModeScope scope2{graph_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{graph_id1};
        {
            ForceBackpropModeScope scope2{graph_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleVariedArgumentTypes) {
    testing::ContextSession context_session{};

    GraphScope graph_scope1{"graph1"};
    GraphScope graph_scope2{"graph2"};
    GraphId graph_id1 = graph_scope1.graph_id();
    GraphId graph_id2 = graph_scope2.graph_id();

    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_FALSE(IsBackpropRequired(graph_id2));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_FALSE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_FALSE(IsBackpropRequired(graph_id2));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_FALSE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{graph_id1};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_FALSE(IsBackpropRequired(graph_id2));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_FALSE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_FALSE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{graph_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{graph_id1};
        {
            NoBackpropModeScope scope2{};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_FALSE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{graph_id1};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    {
        NoBackpropModeScope scope1{graph_id1};
        {
            NoBackpropModeScope scope2{graph_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleGraphArguments) {
    testing::ContextSession context_session{};

    GraphScope graph_scope1{"graph1"};
    GraphScope graph_scope2{"graph2"};
    GraphId graph_id1 = graph_scope1.graph_id();
    GraphId graph_id2 = graph_scope2.graph_id();

    {
        {
            NoBackpropModeScope scope1{graph_id1, graph_id2};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_FALSE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
    {
        {
            std::vector<GraphId> graph_ids{graph_id1, graph_id2};
            NoBackpropModeScope scope1{graph_ids};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_FALSE(IsBackpropRequired(graph_id2));
        }
        EXPECT_TRUE(IsBackpropRequired());
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id2));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeOneContext) {
    testing::ContextSession context_session{};

    GraphScope graph_scope1{"graph1"};
    GraphId graph_id1 = graph_scope1.graph_id();

    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session.context()));
        {
            ForceBackpropModeScope scope2{graph_id1};
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session.context()));
            EXPECT_TRUE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id1, context_session.context()));
            {
                NoBackpropModeScope scope3{graph_id1};
                EXPECT_FALSE(IsBackpropRequired());
                EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session.context()));
                EXPECT_FALSE(IsBackpropRequired(graph_id1));
                EXPECT_FALSE(IsBackpropRequired(graph_id1, context_session.context()));
            }
            EXPECT_FALSE(IsBackpropRequired());
            EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session.context()));
            EXPECT_TRUE(IsBackpropRequired(graph_id1));
            EXPECT_TRUE(IsBackpropRequired(graph_id1, context_session.context()));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session.context()));
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_FALSE(IsBackpropRequired(graph_id1, context_session.context()));
    }
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleContexts) {
    testing::ContextSession context_session1{};

    GraphScope graph_scope1{"graph1"};
    GraphId graph_id1 = graph_scope1.graph_id();

    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session1.context()));
        {
            // New context stack, and a stack for the context should be created.
            testing::ContextSession context_session2{};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(nonstd::nullopt, context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session1.context()));

            NoBackpropModeScope scope1{graph_id1};
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(nonstd::nullopt, context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_FALSE(IsBackpropRequired(graph_id1, context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session1.context()));
            EXPECT_FALSE(IsBackpropRequired(graph_id1, context_session1.context()));
            {
                ForceBackpropModeScope scope2{graph_id1};
                EXPECT_TRUE(IsBackpropRequired());
                EXPECT_TRUE(IsBackpropRequired(nonstd::nullopt, context_session2.context()));
                EXPECT_TRUE(IsBackpropRequired(graph_id1));
                EXPECT_TRUE(IsBackpropRequired(graph_id1, context_session2.context()));
                EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session1.context()));
                EXPECT_FALSE(IsBackpropRequired(graph_id1, context_session1.context()));
            }
            EXPECT_TRUE(IsBackpropRequired());
            EXPECT_TRUE(IsBackpropRequired(nonstd::nullopt, context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(graph_id1));
            EXPECT_FALSE(IsBackpropRequired(graph_id1, context_session2.context()));
            EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session1.context()));
            EXPECT_FALSE(IsBackpropRequired(graph_id1, context_session1.context()));
        }
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session1.context()));
        EXPECT_FALSE(IsBackpropRequired(graph_id1));
        EXPECT_FALSE(IsBackpropRequired(graph_id1, context_session1.context()));
    }
}

// It is possible to use in flat scope because, in C++ spec, dtors are called in reverse order of ctors.
TEST(BackpropModeScopeTest, BackpropModeScopeFlatScope) {
    testing::ContextSession context_session{};

    GraphScope graph_scope1{"graph1"};
    GraphId graph_id1 = graph_scope1.graph_id();

    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session.context()));

        ForceBackpropModeScope scope2{graph_id1};
        EXPECT_FALSE(IsBackpropRequired());
        EXPECT_FALSE(IsBackpropRequired(nonstd::nullopt, context_session.context()));
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        EXPECT_TRUE(IsBackpropRequired(graph_id1, context_session.context()));
    }
}

TEST(BackpropModeScopeTest, BackpropModeWithoutContext) {
    EXPECT_THROW({ NoBackpropModeScope{}; }, ContextError);
    EXPECT_THROW({ ForceBackpropModeScope{}; }, ContextError);
}

TEST(BackpropModeScopeTest, BackpropModeScopeWithAnotherContext) {
    testing::ContextSession context_session{};

    GraphScope graph_scope1{"graph1"};
    GraphScope graph_scope2{"graph2"};
    GraphId graph_id1 = graph_scope1.graph_id();
    GraphId graph_id2 = graph_scope2.graph_id();

    Context another_context{};
    {
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        {
            NoBackpropModeScope scope1{another_context};
            EXPECT_FALSE(IsBackpropRequired(graph_id1, another_context));
            EXPECT_TRUE(IsBackpropRequired(graph_id1));
        }
    }
    {
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        {
            NoBackpropModeScope scope1{{graph_id1, graph_id2}, another_context};
            EXPECT_FALSE(IsBackpropRequired(graph_id1, another_context));
            EXPECT_TRUE(IsBackpropRequired(graph_id1));
        }
    }
    {
        EXPECT_TRUE(IsBackpropRequired(graph_id1));
        {
            std::vector<GraphId> graph_ids{graph_id1, graph_id2};
            NoBackpropModeScope scope1{graph_ids, another_context};
            EXPECT_FALSE(IsBackpropRequired(graph_id1, another_context));
            EXPECT_TRUE(IsBackpropRequired(graph_id1));
        }
    }
}

TEST(BackpropModeScopeTest, IsGradRequiredNoGraph) {
    testing::DeviceSession device_session{DeviceId{"native", 0}};
    Array a = testing::BuildArray({2, 1}).WithLinearData<float>();

    EXPECT_FALSE(IsGradRequired(a));
    EXPECT_FALSE(IsGradRequired(a, AnyGraph{}));
}

TEST(BackpropModeScopeTest, IsGradRequiredSingleGraph) {
    testing::DeviceSession device_session{DeviceId{"native", 0}};

    GraphScope graph_scope1{"graph1"};
    GraphId graph_id1 = graph_scope1.graph_id();

    Array a = testing::BuildArray({2, 1}).WithLinearData<float>();
    a.RequireGrad(graph_id1);

    EXPECT_FALSE(IsGradRequired(a));
    EXPECT_TRUE(IsGradRequired(a, graph_id1));
    EXPECT_TRUE(IsGradRequired(a, AnyGraph{}));
    {
        NoBackpropModeScope scope1{};
        EXPECT_FALSE(IsGradRequired(a));
        EXPECT_FALSE(IsGradRequired(a, graph_id1));
        EXPECT_FALSE(IsGradRequired(a, AnyGraph{}));
        {
            ForceBackpropModeScope scope2{graph_id1};
            EXPECT_FALSE(IsGradRequired(a));
            EXPECT_TRUE(IsGradRequired(a, graph_id1));
            EXPECT_TRUE(IsGradRequired(a, AnyGraph{}));
        }
    }
}

TEST(BackpropModeScopeTest, IsGradRequiredMultipleGraphs) {
    testing::DeviceSession device_session{DeviceId{"native", 0}};

    GraphScope graph_scope1{"graph1"};
    GraphScope graph_scope2{"graph2"};
    GraphScope graph_scope3{"graph3"};
    GraphId graph_id1 = graph_scope1.graph_id();
    GraphId graph_id2 = graph_scope2.graph_id();
    GraphId graph_id3 = graph_scope3.graph_id();

    Array a = testing::BuildArray({2, 1}).WithLinearData<float>();
    a.RequireGrad(graph_id1);
    a.RequireGrad(graph_id2);

    EXPECT_TRUE(IsGradRequired(a, graph_id1));
    EXPECT_TRUE(IsGradRequired(a, graph_id2));
    EXPECT_TRUE(IsGradRequired(a, AnyGraph{}));
    {
        NoBackpropModeScope scope1{graph_id1};
        EXPECT_FALSE(IsGradRequired(a, graph_id1));
        EXPECT_TRUE(IsGradRequired(a, graph_id2));
        EXPECT_TRUE(IsGradRequired(a, AnyGraph{}));
        {
            NoBackpropModeScope scope2{graph_id2};
            EXPECT_FALSE(IsGradRequired(a, graph_id1));
            EXPECT_FALSE(IsGradRequired(a, graph_id2));
            EXPECT_FALSE(IsGradRequired(a, AnyGraph{}));
            {
                ForceBackpropModeScope scope3{graph_id1};
                EXPECT_TRUE(IsGradRequired(a, graph_id1));
                EXPECT_FALSE(IsGradRequired(a, graph_id2));
                EXPECT_TRUE(IsGradRequired(a, AnyGraph{}));
            }
            {
                ForceBackpropModeScope scope3{graph_id2};
                EXPECT_FALSE(IsGradRequired(a, graph_id1));
                EXPECT_TRUE(IsGradRequired(a, graph_id2));
                EXPECT_TRUE(IsGradRequired(a, AnyGraph{}));
            }
            {
                ForceBackpropModeScope scope3{{graph_id3}};
                EXPECT_FALSE(IsGradRequired(a, graph_id1));
                EXPECT_FALSE(IsGradRequired(a, graph_id2));
                EXPECT_FALSE(IsGradRequired(a, AnyGraph{}));
            }
        }
    }
    {
        NoBackpropModeScope scope{};
        EXPECT_FALSE(IsGradRequired(a, graph_id1));
        EXPECT_FALSE(IsGradRequired(a, graph_id2));
        EXPECT_FALSE(IsGradRequired(a, AnyGraph{}));
    }
    {
        NoBackpropModeScope scope{graph_id1, graph_id2};
        EXPECT_FALSE(IsGradRequired(a, graph_id1));
        EXPECT_FALSE(IsGradRequired(a, graph_id2));
        EXPECT_FALSE(IsGradRequired(a, AnyGraph{}));
    }
}

TEST(BackpropModeScopeTest, IsGradRequiredAnotherContext) {
    testing::DeviceSession device_session{DeviceId{"native", 0}};

    GraphScope graph_scope1{"graph1"};
    GraphId graph_id1 = graph_scope1.graph_id();

    Array a = testing::BuildArray({2, 1}).WithLinearData<float>();
    a.RequireGrad(graph_id1);

    EXPECT_FALSE(IsGradRequired(a));
    EXPECT_TRUE(IsGradRequired(a, graph_id1));
    EXPECT_TRUE(IsGradRequired(a, AnyGraph{}));
    {
        testing::ContextSession another_context_session{};
        NoBackpropModeScope scope{};
        // BackpropModeScope of another context does not reflect.
        EXPECT_FALSE(IsGradRequired(a));
        EXPECT_TRUE(IsGradRequired(a, graph_id1));
        EXPECT_TRUE(IsGradRequired(a, AnyGraph{}));
    }
}

}  // namespace
}  // namespace xchainer
