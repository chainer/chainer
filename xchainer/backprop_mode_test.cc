#include "xchainer/backprop_mode.h"

#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/context.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"

namespace xchainer {
namespace {

void ExpectLastBackpropModeEqual(Context& context, const nonstd::optional<GraphId>& graph_id, bool backprop) {
    const internal::BackpropMode& actual = internal::GetBackpropModeStack()->back();
    EXPECT_EQ(&context, &actual.context());
    EXPECT_EQ(graph_id, actual.graph_id());
    EXPECT_EQ(backprop, actual.backprop());
}

TEST(BackpropModeScopeTest, BackpropModeScopeOneContext) {
    Context context1{};
    ContextScope context_scope1{context1};

    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope backprop_mode_scope1{};
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectLastBackpropModeEqual(GetDefaultContext(), nonstd::nullopt, false);
        {
            ForceBackpropModeScope backprop_mode_scope2{"default"};
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectLastBackpropModeEqual(GetDefaultContext(), {"default"}, true);
            {
                NoBackpropModeScope backprop_mode_scope3{"default"};
                EXPECT_EQ(size_t{3}, internal::GetBackpropModeStack()->size());
                ExpectLastBackpropModeEqual(GetDefaultContext(), {"default"}, false);
            }
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectLastBackpropModeEqual(GetDefaultContext(), {"default"}, true);
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectLastBackpropModeEqual(GetDefaultContext(), nonstd::nullopt, false);
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
}

TEST(BackpropModeScopeTest, BackpropModeScopeMultipleContexts) {
    Context context1{};
    ContextScope context_scope1{context1};

    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope backprop_mode_scope1;
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectLastBackpropModeEqual(GetDefaultContext(), nonstd::nullopt, false);
        {
            Context context2{};
            ContextScope context_scope2{context2};
            EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());

            NoBackpropModeScope backprop_mode_scope1{"default"};
            // New context stack, and a stack for the context should be created.
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectLastBackpropModeEqual(GetDefaultContext(), {"default"}, false);
            {
                ForceBackpropModeScope backprop_mode_scope2{"default"};
                EXPECT_EQ(size_t{3}, internal::GetBackpropModeStack()->size());
                ExpectLastBackpropModeEqual(GetDefaultContext(), {"default"}, true);
            }
            EXPECT_EQ(size_t{2}, internal::GetBackpropModeStack()->size());
            ExpectLastBackpropModeEqual(GetDefaultContext(), {"default"}, false);
        }
        EXPECT_EQ(size_t{1}, internal::GetBackpropModeStack()->size());
        ExpectLastBackpropModeEqual(GetDefaultContext(), nonstd::nullopt, false);
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
}

// It is possible to use in flat scope because, in C++ spec, dtors are called in reverse order of ctors.
TEST(BackpropModeScopeTest, BackpropModeScopeFlatScope) {
    Context context1{};
    ContextScope context_scope1{context1};

    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
    {
        NoBackpropModeScope backprop_mode_scope1{};
        ExpectLastBackpropModeEqual(GetDefaultContext(), nonstd::nullopt, false);

        ForceBackpropModeScope backprop_mode_scope2{"default"};
        ExpectLastBackpropModeEqual(GetDefaultContext(), {"default"}, true);
    }
    EXPECT_EQ(nullptr, internal::GetBackpropModeStack());
}

// TODO(niboshi): Write a test where the outermost scope is ForceBackpropMode.

TEST(BackpropModeScopeTest, BackpropModeWithoutContext) {
    EXPECT_THROW({ NoBackpropModeScope{}; }, ContextError);
    EXPECT_THROW({ ForceBackpropModeScope{}; }, ContextError);
}

}  // namespace
}  // namespace xchainer
