#include "xchainer/backprop_mode.h"

#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/testing/context_session.h"

namespace xchainer {
namespace {

void ExpectBackpropModeEqual(size_t i, Context& context, const nonstd::optional<GraphId>& graph_id, bool backprop) {
    const internal::BackpropMode& actual = (*internal::GetBackpropModeStack())[i];
    EXPECT_EQ(&context, &actual.context());
    EXPECT_EQ(graph_id, actual.graph_id());
    EXPECT_EQ(backprop, actual.backprop());
}

TEST(BackpropModeScopeTest, NoBackpropModeScopeSingle) {
    testing::ContextSession context_session{};

    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    EXPECT_TRUE(internal::IsBackpropRequired());
    EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    {
        NoBackpropModeScope scope{};
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
        EXPECT_FALSE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    EXPECT_TRUE(internal::IsBackpropRequired());
    EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    EXPECT_TRUE(internal::IsBackpropRequired("graph2"));
    {
        NoBackpropModeScope scope{{"graph1", "graph2"}};
        EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
        ExpectBackpropModeEqual(1, context_session.context(), {"graph2"}, false);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
        EXPECT_FALSE(internal::IsBackpropRequired("graph2"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    EXPECT_TRUE(internal::IsBackpropRequired());
    EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    EXPECT_TRUE(internal::IsBackpropRequired("graph2"));
    {
        NoBackpropModeScope scope{{}};
        EXPECT_EQ(size_t{0}, internal::GetBackpropModeStack()->size());
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    EXPECT_TRUE(internal::IsBackpropRequired());
    EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
}

TEST(BackpropModeScopeTest, ForceBackpropModeScopeSingle) {
    testing::ContextSession context_session{};

    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    EXPECT_TRUE(internal::IsBackpropRequired());
    EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    {
        ForceBackpropModeScope scope{};
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, true);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    EXPECT_TRUE(internal::IsBackpropRequired());
    EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    EXPECT_TRUE(internal::IsBackpropRequired("graph2"));
    {
        ForceBackpropModeScope scope{{"graph1", "graph2"}};
        EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, true);
        ExpectBackpropModeEqual(1, context_session.context(), {"graph2"}, true);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
        EXPECT_TRUE(internal::IsBackpropRequired("graph2"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    EXPECT_TRUE(internal::IsBackpropRequired());
    EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    EXPECT_TRUE(internal::IsBackpropRequired("graph2"));
    {
        ForceBackpropModeScope scope{{}};
        EXPECT_EQ(size_t{0}, internal::GetBackpropModeStack()->size());
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    EXPECT_TRUE(internal::IsBackpropRequired());
    EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultiple) {
    testing::ContextSession context_session{};

    {
        ForceBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, true);
            ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, false);
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, true);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        ForceBackpropModeScope scope1{{"graph1"}};
        {
            ForceBackpropModeScope scope2{{"graph1"}};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, true);
            ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, true);
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, true);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
            ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, false);
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            ForceBackpropModeScope scope2{{"graph1"}};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
            ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, true);
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleVariedArgumentTypes) {
    testing::ContextSession context_session{};

    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
            ExpectBackpropModeEqual(1, context_session.context(), nonstd::nullopt, false);
            EXPECT_FALSE(internal::IsBackpropRequired());
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
        EXPECT_FALSE(internal::IsBackpropRequired());
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
            EXPECT_FALSE(internal::IsBackpropRequired());
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
        EXPECT_FALSE(internal::IsBackpropRequired());
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
            ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, false);
            EXPECT_FALSE(internal::IsBackpropRequired());
            EXPECT_FALSE(internal::IsBackpropRequired("graph"));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
        EXPECT_FALSE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired("graph"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{};
            EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
            EXPECT_FALSE(internal::IsBackpropRequired());
        }
        EXPECT_EQ(size_t{0}, internal::GetBackpropModeStack()->size());
        EXPECT_TRUE(internal::IsBackpropRequired());
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_EQ(size_t{0}, internal::GetBackpropModeStack()->size());
            EXPECT_TRUE(internal::IsBackpropRequired());
        }
        EXPECT_EQ(size_t{0}, internal::GetBackpropModeStack()->size());
        EXPECT_TRUE(internal::IsBackpropRequired());
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{{}};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
        }
        EXPECT_EQ(size_t{0}, internal::GetBackpropModeStack()->size());
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
            ExpectBackpropModeEqual(1, context_session.context(), nonstd::nullopt, false);
            EXPECT_FALSE(internal::IsBackpropRequired());
            EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{{}};
            EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{{"graph1"}};
        {
            NoBackpropModeScope scope2{{"graph1"}};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
            ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, false);
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), {"graph1"}, false);
        EXPECT_TRUE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
}

TEST(BackpropModeScopeTest, BackpropModeScopeOneContext) {
    testing::ContextSession context_session{};

    {
        NoBackpropModeScope scope1{};
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
        EXPECT_FALSE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session.context()));
        {
            ForceBackpropModeScope scope2{{"graph1"}};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
            ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, true);
            EXPECT_FALSE(internal::IsBackpropRequired());
            EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session.context()));
            EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
            EXPECT_TRUE(internal::IsBackpropRequired("graph1", context_session.context()));
            {
                NoBackpropModeScope scope3{{"graph1"}};
                EXPECT_EQ(size_t{3}, internal::GetBackpropModeStack()->size());
                ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
                ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, true);
                ExpectBackpropModeEqual(2, context_session.context(), {"graph1"}, false);
                EXPECT_FALSE(internal::IsBackpropRequired());
                EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session.context()));
                EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
                EXPECT_FALSE(internal::IsBackpropRequired("graph1", context_session.context()));
            }
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
            ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, true);
            EXPECT_FALSE(internal::IsBackpropRequired());
            EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session.context()));
            EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
            EXPECT_TRUE(internal::IsBackpropRequired("graph1", context_session.context()));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
        EXPECT_FALSE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session.context()));
        EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
        EXPECT_FALSE(internal::IsBackpropRequired("graph1", context_session.context()));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleContexts) {
    testing::ContextSession context_session1{};

    {
        NoBackpropModeScope scope1{};
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session1.context(), nonstd::nullopt, false);
        EXPECT_FALSE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session1.context()));
        {
            // New context stack, and a stack for the context should be created.
            testing::ContextSession context_session2{};
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_TRUE(internal::IsBackpropRequired(kDefaultGraphId, context_session2.context()));
            EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session1.context()));

            NoBackpropModeScope scope1{{"graph1"}};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session1.context(), nonstd::nullopt, false);
            ExpectBackpropModeEqual(1, context_session2.context(), {"graph1"}, false);
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_TRUE(internal::IsBackpropRequired(kDefaultGraphId, context_session2.context()));
            EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
            EXPECT_FALSE(internal::IsBackpropRequired("graph1", context_session2.context()));
            EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session1.context()));
            EXPECT_FALSE(internal::IsBackpropRequired("graph1", context_session1.context()));
            {
                ForceBackpropModeScope scope2{{"graph1"}};
                EXPECT_EQ(size_t{3}, internal::GetBackpropModeStack()->size());
                ExpectBackpropModeEqual(0, context_session1.context(), nonstd::nullopt, false);
                ExpectBackpropModeEqual(1, context_session2.context(), {"graph1"}, false);
                ExpectBackpropModeEqual(2, context_session2.context(), {"graph1"}, true);
                EXPECT_TRUE(internal::IsBackpropRequired());
                EXPECT_TRUE(internal::IsBackpropRequired(kDefaultGraphId, context_session2.context()));
                EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
                EXPECT_TRUE(internal::IsBackpropRequired("graph1", context_session2.context()));
                EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session1.context()));
                EXPECT_FALSE(internal::IsBackpropRequired("graph1", context_session1.context()));
            }
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectBackpropModeEqual(0, context_session1.context(), nonstd::nullopt, false);
            ExpectBackpropModeEqual(1, context_session2.context(), {"graph1"}, false);
            EXPECT_TRUE(internal::IsBackpropRequired());
            EXPECT_TRUE(internal::IsBackpropRequired(kDefaultGraphId, context_session2.context()));
            EXPECT_FALSE(internal::IsBackpropRequired("graph1"));
            EXPECT_FALSE(internal::IsBackpropRequired("graph1", context_session2.context()));
            EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session1.context()));
            EXPECT_FALSE(internal::IsBackpropRequired("graph1", context_session1.context()));
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectBackpropModeEqual(0, context_session1.context(), nonstd::nullopt, false);
        EXPECT_FALSE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session1.context()));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
}

// It is possible to use in flat scope because, in C++ spec, dtors are called in reverse order of ctors.
TEST(BackpropModeScopeTest, BackpropModeScopeFlatScope) {
    testing::ContextSession context_session{};

    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope scope1{};
        ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
        EXPECT_FALSE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session.context()));

        ForceBackpropModeScope scope2{{"graph1"}};
        ExpectBackpropModeEqual(0, context_session.context(), nonstd::nullopt, false);
        ExpectBackpropModeEqual(1, context_session.context(), {"graph1"}, true);
        EXPECT_FALSE(internal::IsBackpropRequired());
        EXPECT_FALSE(internal::IsBackpropRequired(kDefaultGraphId, context_session.context()));
        EXPECT_TRUE(internal::IsBackpropRequired("graph1"));
        EXPECT_TRUE(internal::IsBackpropRequired("graph1", context_session.context()));
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
}

TEST(BackpropModeScopeTest, BackpropModeWithoutContext) {
    EXPECT_THROW({ NoBackpropModeScope{}; }, ContextError);
    EXPECT_THROW({ ForceBackpropModeScope{}; }, ContextError);
}

}  // namespace
}  // namespace xchainer
