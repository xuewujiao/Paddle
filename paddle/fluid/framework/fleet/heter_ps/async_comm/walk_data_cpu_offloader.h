#pragma once

#include <cuda_runtime_api.h>

class WalkDataCPUOffloader {
 public:
  WalkDataCPUOffloader() = delete;
  WalkDataCPUOffloader(const WalkDataCPUOffloader&) = delete;
  WalkDataCPUOffloader(WalkDataCPUOffloader&&) = delete;
  WalkDataCPUOffloader(int max_walk_count, int max_node_count, int element_byte, cudaStream_t stream);
  ~WalkDataCPUOffloader();
  void StartOffloadingSteps(int node_count);
  void StartSingleStepOffload(const void* data_d, int walk_id);
  void WaitLastOffloadStepDone();
  void GetOffloadedData(void* data_d, int start_node_idx, int node_count);
  void StartAsyncGetOffloadedData(void* data_d, int start_node_idx, int node_count);
  void WaitAsyncGetOffloadedDataDone();
 private:
  int max_walk_count_ = 0;
  int max_node_count_ = 0;
  int element_byte_ = sizeof(int64_t);
  int node_count_ = 0;
  int walk_id_ = 0;
  char* data_h_ = 0;
  cudaStream_t stream_ = nullptr;
};
