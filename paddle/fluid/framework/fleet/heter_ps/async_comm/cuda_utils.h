#pragma once

#include <cuda_runtime_api.h>

void EnableAllPeerAccess();

cudaStream_t CreateHighPriorityStream(int flags);

void MemcpyUnique(void* dst, const void* src, size_t size, cudaStream_t stream);
