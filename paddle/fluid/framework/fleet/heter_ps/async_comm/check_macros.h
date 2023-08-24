#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t const status = call;                                           \
    if (status != cudaSuccess) {                                               \
      cudaGetLastError();                                                      \
      fprintf(stderr, "CUDA error encountered at: "                            \
                                "file=%s, line=%d, "                           \
                                "call='%s', Reason=%s:%s",                     \
                                __FILE__, __LINE__,                            \
                                #call,                                         \
                                cudaGetErrorName(status),                      \
                                cudaGetErrorString(status));                   \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CU_CHECK(call)                                                         \
  do {                                                                         \
    auto result = call;                                                        \
    if (result != CUDA_SUCCESS) {                                              \
      const char *p_err_str = nullptr;                                         \
      if (cuGetErrorString(result, &p_err_str) == CUDA_ERROR_INVALID_VALUE) {  \
        p_err_str = "Unrecoginzed CU error num";                               \
      }                                                                        \
      fprintf(stderr, "CU error encountered at: "                              \
              "file=%s line=%d, call='%s' Reason=%s.\n",                       \
              __FILE__, __LINE__,                                              \
              #call, p_err_str);                                               \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CALL_CHECK(call)                                                       \
  do {                                                                         \
    int result = call;                                                         \
    if (result != 0) {                                                         \
      fprintf(stderr, "file=%s, line=%d, call='%s', returned=%d.\n",           \
              __FILE__, __LINE__, #call, result);                              \
      abort();                                                                 \
    }                                                                          \
  } while(0)

#define BOOL_CHECK(call)                                                       \
  do {                                                                         \
    bool result = call;                                                        \
    if (result != true) {                                                      \
      fprintf(stderr, "file=%s, line=%d, expression='%s', evaluated false.\n", \
              __FILE__, __LINE__, #call);                                      \
      abort();                                                                 \
    }                                                                          \
  } while(0)
