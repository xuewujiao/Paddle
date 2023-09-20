#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>

#include "message_queue.h"
#include "meta.h"

class AsyncReqRes;
class Partitioner;
class AsyncCommunicator;
class MemoryAllocatorBase;
class IntraNodeCommunicator;

#define MAX_GPUS_PER_NODE (8)

struct MergedRankInfo {
  size_t request_offset[MAX_VEC_COUNT];
  size_t request_size[MAX_VEC_COUNT];
  size_t response_offset[MAX_VEC_COUNT];
  size_t response_size[MAX_VEC_COUNT];
};
struct MergedResponseInfo {
  int location = 0;
  int dtype = 0;
  size_t data_size = 0;
};

struct MergeHeader {
  MergedRankInfo ranks_info[MAX_GPUS_PER_NODE];
  MergedResponseInfo response_info[MAX_VEC_COUNT];
  int response_data_count;
};

struct MergedHandle {
  AsyncReqRes* request = nullptr;
  AsyncReqRes* response = nullptr;
  AsyncReqRes* splitted_requests[MAX_GPUS_PER_NODE];
  AsyncReqRes* splitted_responses[MAX_GPUS_PER_NODE];
  std::mutex mutex{};
  int pending_count = 0;
  std::function<void(MergedHandle*)> done_cb = nullptr;

  void FinishRequest() {
    std::unique_lock<std::mutex> mlock(mutex);
    pending_count--;
    if (pending_count == 0) {
      done_cb(this);
    }
  }
};

class MergedHandler {
 public:
  explicit MergedHandler(Partitioner* partitioner, MemoryAllocatorBase* allocator, AsyncCommunicator* async_communicator);
  ~MergedHandler();
  void Start();
  void Stop();
  void WaitStopped();
  void PutRequestAsync(AsyncReqRes* request);
  AsyncReqRes* OnRunnerGetResponse(Meta* meta);
  AsyncReqRes* OnReceiveGetResponse(Meta* meta);
  void OnReceiveResponse(AsyncReqRes* response);
 private:
  void DispatchThreadFunc();
  void FreeThreadFunc();
  MemoryAllocatorBase* allocator_ = nullptr;
  Partitioner* partitioner_ = nullptr;
  AsyncCommunicator* async_communicator_ = nullptr;
  MessageQueue<AsyncReqRes*> request_message_queue_;
  std::unique_ptr<std::thread> dispatch_thread_;

  MessageQueue<AsyncReqRes*> to_free_queue_;
  std::unique_ptr<std::thread> free_thread_;

  std::mutex mutex_{};
  std::unordered_map<uint64_t, MergedHandle*> request_map_;
  std::atomic<int> pending_frees_{};
};