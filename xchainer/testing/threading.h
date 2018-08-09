#pragma once

#include <condition_variable>
#include <cstddef>
#include <thread>
#include <vector>

namespace xchainer {
namespace testing {

template <typename Func, typename... Args>
auto RunThreads(size_t thread_count, const Func& func, Args&&... args) -> std::vector<decltype(func(size_t{}, std::declval<Args>()...))> {
    using ResultType = decltype(func(size_t{}, std::declval<Args>()...));

    std::mutex mutex;
    std::condition_variable cv_all_ready;

    std::vector<ResultType> results;
    results.resize(thread_count);

    size_t wait_count = thread_count;

    auto thread_proc = [&mutex, &cv_all_ready, &wait_count, &func, &results, &args...](size_t thread_index) mutable {
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

        ResultType result = func(thread_index, std::forward<Args>(args)...);

        {
            std::lock_guard<std::mutex> lock{mutex};
            gsl::at(results, thread_index) = std::move(result);
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        threads.emplace_back(thread_proc, i);
    }

    // Join threads
    for (std::thread& thread : threads) {
        thread.join();
    }

    return results;
}

template <typename SetupFunc, typename Func, typename CheckFunc>
void CheckThreadSafety(size_t thread_count, size_t repeat_count, const SetupFunc& setupFunc, const Func& func, const CheckFunc& checkFunc) {
    using CheckContextType = decltype(setupFunc(size_t{}));
    using ResultType = decltype(func(size_t{}, std::declval<CheckContextType>()));

    for (size_t i_repeat = 0; i_repeat < repeat_count; ++i_repeat) {
        CheckContextType checkContext = setupFunc(i_repeat);

        std::vector<ResultType> results = RunThreads(thread_count, func, checkContext);

        checkFunc(results);
    }
}

}  // namespace testing
}  // namespace xchainer
