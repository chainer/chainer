#include "chainerx/thread_local_state.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include "chainerx/backprop_mode.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/testing/threading.h"

// In the following tests, main test logic of each test case is run on a different thread than the main thread (where the test is invoked),
// because the thread local storage of the main thread may be dirty.

namespace chainerx {
namespace {

TEST(ThreadLocalStateTest, SingleThreadGetAndSet) {
    std::thread thread{[]() {
        Context context{};
        Device& device = context.GetDevice({"native", 0});

        // Set some thread local variables
        ThreadLocalState state{};
        {
            ContextScope context_scope{context};
            DeviceScope device_scope{device};
            NoBackpropModeScope no_backprop_mode_scope{};

            // Save state
            state = ThreadLocalState::Get();
        }

        ASSERT_EQ(nullptr, internal::GetDefaultContextNoExcept());
        ASSERT_EQ(nullptr, internal::GetDefaultDeviceNoExcept());
        ASSERT_TRUE(IsBackpropRequired(context.default_backprop_id()));

        // Restore state
        ThreadLocalState::Set(state);

        EXPECT_EQ(&context, internal::GetDefaultContextNoExcept());
        EXPECT_EQ(&device, internal::GetDefaultDeviceNoExcept());
        EXPECT_FALSE(IsBackpropRequired(context.default_backprop_id()));
    }};
    thread.join();
}

TEST(ThreadLocalStateTest, Copy) {
    std::thread thread{[]() {
        Context context{};
        Device& device = context.GetDevice({"native", 0});

        // Set some thread local variables
        ThreadLocalState state2{};
        {
            ContextScope context_scope{context};
            DeviceScope device_scope{device};
            NoBackpropModeScope no_backprop_mode_scope{};

            // Save state
            ThreadLocalState state = ThreadLocalState::Get();

            // Copy
            state2 = state;
        }

        ASSERT_EQ(nullptr, internal::GetDefaultContextNoExcept());
        ASSERT_EQ(nullptr, internal::GetDefaultDeviceNoExcept());
        ASSERT_TRUE(IsBackpropRequired(context.default_backprop_id()));

        // Restore state
        ThreadLocalState::Set(state2);

        EXPECT_EQ(&context, internal::GetDefaultContextNoExcept());
        EXPECT_EQ(&device, internal::GetDefaultDeviceNoExcept());
        EXPECT_FALSE(IsBackpropRequired(context.default_backprop_id()));
    }};
    thread.join();
}

TEST(ThreadLocalStateTest, Move) {
    std::thread thread{[]() {
        Context context{};
        Device& device = context.GetDevice({"native", 0});

        // Set some thread local variables
        ThreadLocalState state2{};
        {
            ContextScope context_scope{context};
            DeviceScope device_scope{device};
            NoBackpropModeScope no_backprop_mode_scope{};

            // Save state
            ThreadLocalState state = ThreadLocalState::Get();

            // Move
            state2 = std::move(state);
        }

        ASSERT_EQ(nullptr, internal::GetDefaultContextNoExcept());
        ASSERT_EQ(nullptr, internal::GetDefaultDeviceNoExcept());
        ASSERT_TRUE(IsBackpropRequired(context.default_backprop_id()));

        // Restore state
        ThreadLocalState::Set(state2);

        EXPECT_EQ(&context, internal::GetDefaultContextNoExcept());
        EXPECT_EQ(&device, internal::GetDefaultDeviceNoExcept());
        EXPECT_FALSE(IsBackpropRequired(context.default_backprop_id()));
    }};
    thread.join();
}

TEST(ThreadLocalStateTest, CopyAcrossThreads) {
    std::thread thread{[]() {
        Context context{};
        Device& device = context.GetDevice({"native", 0});
        BackpropId backprop_id_main = context.MakeBackpropId("main");
        BackpropId backprop_id_thread1 = context.MakeBackpropId("thread1");
        BackpropId backprop_id_thread2 = context.MakeBackpropId("thread2");

        // Set some thread local variables
        ThreadLocalState state{};
        {
            ContextScope context_scope{context};
            DeviceScope device_scope{device};
            NoBackpropModeScope no_backprop_mode_scope{backprop_id_main};

            // Save state
            state = ThreadLocalState::Get();

            testing::RunThreads(
                    2U, [state, &context, &device, backprop_id_main, backprop_id_thread1, backprop_id_thread2](size_t thread_index) {
                        // Restore state
                        ThreadLocalState::Set(state);

                        // Set different backprop mode in each thread
                        BackpropId backprop_id_this_thread = thread_index == 0 ? backprop_id_thread1 : backprop_id_thread2;
                        BackpropId backprop_id_another_thread = thread_index == 0 ? backprop_id_thread2 : backprop_id_thread1;
                        NoBackpropModeScope no_backprop_mode_scope_thread{backprop_id_this_thread};

                        EXPECT_EQ(&context, internal::GetDefaultContextNoExcept());
                        EXPECT_EQ(&device, internal::GetDefaultDeviceNoExcept());
                        EXPECT_FALSE(IsBackpropRequired(backprop_id_main));
                        EXPECT_FALSE(IsBackpropRequired(backprop_id_this_thread));
                        EXPECT_TRUE(IsBackpropRequired(backprop_id_another_thread));
                    });
        }
    }};
    thread.join();
}

TEST(ThreadLocalStateTest, SourceThreadStateInvalidated) {
    std::thread source_thread{[]() {
        Context context{};
        Device& device = context.GetDevice({"native", 0});
        std::mutex mutex{};
        std::condition_variable cv{};
        bool source_state_invalidated{false};

        // Destination thread
        auto dest_thread_func = [&mutex, &cv, &source_state_invalidated, &context, &device](ThreadLocalState state) mutable {
            {
                // Wait for the source thread to exit the state scope
                std::unique_lock<std::mutex> lock{mutex};
                cv.wait(lock, [&]() { return source_state_invalidated; });
            }

            ASSERT_EQ(nullptr, internal::GetDefaultContextNoExcept());
            ASSERT_EQ(nullptr, internal::GetDefaultDeviceNoExcept());
            ASSERT_TRUE(IsBackpropRequired(context.default_backprop_id()));

            // Restore state
            ThreadLocalState::Set(state);

            EXPECT_EQ(&context, internal::GetDefaultContextNoExcept());
            EXPECT_EQ(&device, internal::GetDefaultDeviceNoExcept());
            EXPECT_FALSE(IsBackpropRequired(context.default_backprop_id()));
        };

        std::thread dest_thread{};

        // State scope
        {
            ContextScope context_scope{context};
            DeviceScope device_scope{device};
            NoBackpropModeScope no_backprop_mode_scope{context.default_backprop_id()};

            // Save state
            ThreadLocalState state = ThreadLocalState::Get();
            // Launch the destination thread
            dest_thread = std::thread{dest_thread_func, std::move(state)};
        }

        // Notify the destination thread
        {
            std::unique_lock<std::mutex> lock{mutex};
            source_state_invalidated = true;
            cv.notify_all();
        }

        dest_thread.join();
    }};

    source_thread.join();
}

}  // namespace
}  // namespace chainerx
