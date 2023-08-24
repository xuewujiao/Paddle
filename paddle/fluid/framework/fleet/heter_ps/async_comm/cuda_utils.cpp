#include "cuda_utils.h"

#include "check_macros.h"
#include "log_macros.h"

void EnableAllPeerAccess() {
  int old_dev;
  CUDA_CHECK(cudaGetDevice(&old_dev));
  int dev_count;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));
  BOOL_CHECK(dev_count > 0);
  for (int i = 0; i < dev_count; i++) {
    CUDA_CHECK(cudaSetDevice(i));
    for (int j = 0; j < dev_count; j++) {
      if (i == j) continue;
      int can_p2p = 0;
      CUDA_CHECK(cudaDeviceCanAccessPeer(&can_p2p, i, j));
      if (can_p2p == 1) {
        CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
        LOG_INFO("device %d enabling peer access to %d", i, j);
      }
    }
  }
  CUDA_CHECK(cudaSetDevice(old_dev));
}

cudaStream_t CreateHighPriorityStream(int flags) {
  int least_priority, greatest_priority;
  int dev_id;
  CUDA_CHECK(cudaGetDevice(&dev_id));
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream, flags, greatest_priority));
  LOG_DEBUG("Creating stream with priority %d [H: %d, L: %d] on device %d",
            greatest_priority, greatest_priority, least_priority, dev_id);
  return stream;
}

void MemcpyUnique(void* dst, const void* src, size_t size, cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream));
}