#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <type_traits>
#include <vector>

namespace chainerx {
namespace testing {
namespace threading_detail {

// Called by thread dispatcher.
template <typename Func>
auto CallFunc(const Func& func, size_t thread_index) -> decltype(func(size_t{})) {
    return func(thread_index);
}

// Called by thread dispatcher.
template <typename Func>
auto CallFunc(const Func& func, size_t /*thread_index*/) -> decltype(func()) {
    return func();
}

}  // namespace threading_detail

template <
        typename Func,
        typename ResultType = decltype(threading_detail::CallFunc(std::declval<Func>(), size_t{})),
        std::enable_if_t<!std::is_void<ResultType>::value, std::nullptr_t> = nullptr>
std::vector<ResultType> RunThreads(size_t thread_count, const Func& func) {
    std::mutex mutex;
    std::condition_variable cv_all_ready;

    size_t wait_count = thread_count;

    auto thread_proc = [&mutex, &cv_all_ready, &wait_count, &func](size_t thread_index) mutable -> ResultType {
        {
            std::unique_lock<std::mutex> lock{mutex};

            --wait_count;

            if (wait_count == 0) {
                // If this thread is the last one to be ready, wake up all other threads.
                cv_all_ready.notify_all();
            } else {
                // Otherwise, wait for all other threads to be ready.
                cv_all_ready.wait(lock);
            }
        }

        return threading_detail::CallFunc(func, thread_index);
    };

    // Launch threads
    std::vector<std::future<ResultType>> futures;
    futures.reserve(thread_count);

    for (size_t i = 0; i < thread_count; ++i) {
        futures.emplace_back(std::async(std::launch::async, thread_proc, i));
    }

    // Retrieve results
    std::vector<ResultType> results;
    results.reserve(thread_count);
    std::transform(
            futures.begin(), futures.end(), std::back_inserter(results), [](std::future<ResultType>& future) { return future.get(); });

    return results;
}

template <
        typename Func,
        typename ResultType = decltype(threading_detail::CallFunc(std::declval<Func>(), size_t{})),
        std::enable_if_t<std::is_void<ResultType>::value, std::nullptr_t> = nullptr>
void RunThreads(size_t thread_count, const Func& func) {
    // Call overload by wrapping the given function that returns void with a lambda that returns a nullptr.
    RunThreads(thread_count, [&func](size_t thread_index) {
        threading_detail::CallFunc(func, thread_index);
        return nullptr;
    });
}

// TODO(sonots): Reconsider the function name.
// TODO(sonots): Do single-shot and multi-threads tests in seperated test-cases.
// TODO(sonots): Make it possible to use different contexts and/or devices in different threads.
template <typename Func>
inline void RunTestWithThreads(const Func& func, size_t thread_count = 2) {
    // Run single-shot
    threading_detail::CallFunc(func, -1);

    // Run in multi-threads
    if (thread_count > 0) {
        Context& context = chainerx::GetDefaultContext();
        Device& device = chainerx::GetDefaultDevice();
        RunThreads(thread_count, [&context, &device, &func](size_t thread_index) {
            chainerx::SetDefaultContext(&context);
            chainerx::SetDefaultDevice(&device);
            threading_detail::CallFunc(func, thread_index);
        });
    }
}

#define CHAINERX_TEST_BASE_CLASS_NAME_(test_case_name, test_name) test_case_name##_##test_name##_Base
#define CHAINERX_TEST_SINGLE_THREAD_CLASS_NAME_(test_case_name, test_name) test_case_name##_##test_name##_SingleThread
#define CHAINERX_TEST_MULTI_THREAD_CLASS_NAME_(test_case_name, test_name) test_case_name##_##test_name##_MultiThread

#define TEST_THREAD_SAFE(test_case_name, test_name)                                                             \
    class CHAINERX_TEST_BASE_CLASS_NAME_(test_case_name, test_name) : public ::testing::Test {                  \
    protected:                                                                                                  \
        void RunTestBody();                                                                                     \
                                                                                                                \
    private:                                                                                                    \
        virtual size_t thread_count() = 0;                                                                      \
    };                                                                                                          \
                                                                                                                \
    class CHAINERX_TEST_SINGLE_THREAD_CLASS_NAME_(test_case_name, test_name)                                    \
        : public CHAINERX_TEST_BASE_CLASS_NAME_(test_case_name, test_name) {                                    \
    private:                                                                                                    \
        size_t thread_count() override { return 1; }                                                            \
    };                                                                                                          \
                                                                                                                \
    class CHAINERX_TEST_MULTI_THREAD_CLASS_NAME_(test_case_name, test_name)                                     \
        : public CHAINERX_TEST_BASE_CLASS_NAME_(test_case_name, test_name) {                                    \
    private:                                                                                                    \
        size_t thread_count() override { return 2; }                                                            \
    };                                                                                                          \
                                                                                                                \
    TEST_F(CHAINERX_TEST_SINGLE_THREAD_CLASS_NAME_(test_case_name, test_name), SingleThread) { RunTestBody(); } \
                                                                                                                \
    TEST_F(CHAINERX_TEST_MULTI_THREAD_CLASS_NAME_(test_case_name, test_name), MultiThread) { RunTestBody(); }   \
                                                                                                                \
    void CHAINERX_TEST_BASE_CLASS_NAME_(test_case_name, test_name)::RunTestBody()

#define TASK(...)                                                          \
    if (thread_count() > 1) {                                              \
        testing::RunThreads(thread_count(), [&](size_t /*thread_index*/) { \
            { __VA_ARGS__ }                                                \
            return nullptr;                                                \
        });                                                                \
    } else {                                                               \
        { __VA_ARGS__ }                                                    \
    }

}  // namespace testing
}  // namespace chainerx
