#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <vector>

namespace xchainer {
namespace testing {

template <typename Func, typename... Args>
auto RunThreads(size_t thread_count, const Func& func, Args&&... args) -> std::vector<decltype(func(size_t{}, std::declval<Args>()...))> {
    using ResultType = decltype(func(size_t{}, std::declval<Args>()...));

    std::mutex mutex;
    std::condition_variable cv_all_ready;

    size_t wait_count = thread_count;

    auto thread_proc = [&mutex, &cv_all_ready, &wait_count, &func, &args...](size_t thread_index) mutable -> ResultType {
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

        return func(thread_index, args...);
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

template <typename SetupFunc, typename Func, typename CheckFunc>
void CheckThreadSafety(
        size_t repeat_count, size_t thread_count, const SetupFunc& setup_func, const Func& func, const CheckFunc& check_func) {
    using CheckContextType = decltype(setup_func(size_t{}));
    using ResultType = decltype(func(size_t{}, std::declval<CheckContextType>()));

    for (size_t i_repeat = 0; i_repeat < repeat_count; ++i_repeat) {
        CheckContextType check_context = setup_func(i_repeat);

        std::vector<ResultType> results = RunThreads(thread_count, func, check_context);

        check_func(results);
    }
}

// TODO(sonots): Recondier the function name.
// TODO(sonots): Do single-shot and multi-threads tests in seperated test-cases.
// TODO(sonots): Make it possible to use another context and device in another thread.
inline void RunTestWithThreads(const std::function<void(void)>& func, size_t thread_count = 2) {
    // Run single-shot
    func();

    // Run in multi-threads
    if (thread_count > 0) {
        Context& context = xchainer::GetDefaultContext();
        Device& device = xchainer::GetDefaultDevice();
        RunThreads(thread_count, [&context, &device, &func](size_t /*thread_index*/) {
            xchainer::SetDefaultContext(&context);
            xchainer::SetDefaultDevice(&device);
            func();
            return nullptr;
        });
    }
}

}  // namespace testing
}  // namespace xchainer
