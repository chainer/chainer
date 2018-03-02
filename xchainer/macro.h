#pragma once

#ifndef XCHAINER_HOST_DEVICE
#ifdef __CUDACC__
#define XCHAINER_HOST_DEVICE __host__ __device__
#else
#define XCHAINER_HOST_DEVICE
#endif  // __CUDACC__
#endif  // XCHAINER_HOST_DEVICE
