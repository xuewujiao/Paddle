#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "memory_allocator.h"
#include "partitioner.h"

struct AsyncReqRes;
class AsyncCommunicator;

class IntraNodeCommunicator {
 public:
  IntraNodeCommunicator(Partitioner* partitioner, MemoryAllocatorBase* allocator);
  ~IntraNodeCommunicator();

  void CreateResources();
  void DestroyResources();
  void ConnectFifos();
  void Start();
  void Stop();
  void WaitStopped();

  void SetAsyncCommunicator(AsyncCommunicator* async_communicator) {
    async_communicator_ = async_communicator;
  }

  void Send(AsyncReqRes* request);
 private:
  static std::string GetFifoNameForIntraNodeCommunicator(const std::string& type, int node_id, int local_rank);

  void RequestProcessLoop();
  void ResponseProcessLoop();

  int req_read_fifo_ = -1;
  int res_read_fifo_ = -1;
  std::vector<int> req_write_fifo_;
  std::vector<int> res_write_fifo_;

  std::unique_ptr<std::thread> req_thread_;
  std::unique_ptr<std::thread> res_thread_;

  Partitioner* partitioner_;
  MemoryAllocatorBase* allocator_base_;
  AsyncCommunicator* async_communicator_ = nullptr;

  std::atomic<bool> stopped_{};
  std::atomic<int64_t> need_del_req_res_{};
};