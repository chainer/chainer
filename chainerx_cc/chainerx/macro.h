#pragma once

#include <cassert>
#include <cstdlib>

#ifdef NDEBUG
#define CHAINERX_DEBUG false
#else  // NDEBUG
#define CHAINERX_DEBUG true
#endif  // NDEBUG

#if CHAINERX_DEBUG
#define CHAINERX_ASSERT assert
#else  // CHAINERX_DEBUG
// This expression suppresses dead code and unused variable warnings of clang-tidy caused by "unused" __VA_ARGS__.
// We use a lambda call to bypass clant-tidy's dead-code analysis, which currently does not evaluate non-const expressions.
#define CHAINERX_ASSERT(...) (void)([] { return false; }() && (__VA_ARGS__))
#endif  // CHAINERX_DEBUG

#ifndef CHAINERX_HOST_DEVICE
#ifdef __CUDACC__
#define CHAINERX_HOST_DEVICE __host__ __device__
#else  // __CUDA__
#define CHAINERX_HOST_DEVICE
#endif  // __CUDACC__
#endif  // CHAINERX_HOST_DEVICE

#ifndef CHAINERX_NEVER_REACH
#ifdef NDEBUG
#define CHAINERX_NEVER_REACH() (std::abort())
#else  // NDEBUG
#define CHAINERX_NEVER_REACH()                                        \
    do {                                                              \
        assert(false); /* NOLINT(cert-dcl03-c, misc-static-assert) */ \
        std::abort();                                                 \
    } while (false)
#endif  // NDEBUG
#endif  // CHAINERX_NEVER_REACH
