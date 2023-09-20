#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "agent.h"
#include "config.h"
#include "copy_thread.h"
#include "memory_allocator.h"
#include "partitioner.h"

struct AsyncReqRes;
class AsyncCommunicator;

class InterNodeCommunicator {
 public:
  InterNodeCommunicator(Partitioner* partitioner, MemoryAllocatorBase* allocator, Config* config);
  ~InterNodeCommunicator();
  void CreateResources();
  void DestroyResources();
  void Connect();
  void Start();
  void Stop();
  void WaitStopped();

  void SetAsyncCommunicator(AsyncCommunicator* async_communicator) {
    async_communicator_ = async_communicator;
  }

  void Send(AsyncReqRes* req_res);
 private:
  Partitioner* partitioner_;
  MemoryAllocatorBase* allocator_;
  Config* config_;  
  AsyncCommunicator* async_communicator_ = nullptr;

  std::unique_ptr<CopyThread> copy_thread_;
  std::unique_ptr<Agent> agent_;
};