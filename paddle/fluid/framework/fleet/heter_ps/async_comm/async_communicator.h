#pragma once

#include <pthread.h>

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "intranode_communicator.h"
#include "internode_communicator.h"
#include "meta.h"
#include "memory_allocator.h"
#include "merged_handler.h"
#include "message_queue.h"
#include "partitioner.h"
#include "runner.h"
#include "sideband_communicator.h"
#include "glog/logging.h"

class RequestHandle;

class AsyncReqRes {
 public:
  ~AsyncReqRes();
  Meta meta;
  MemoryContextBase* memory_contexts[MAX_VEC_COUNT];
  void FillMetaByMemoryContext();
 private:
  std::function<void()> complete_cb = nullptr;
  AsyncReqRes();
  void Init();
  friend AsyncReqRes *CreateAsyncReqRes();
  friend class AsyncCommunicator;
  friend class MemoryAllocatorBase;
};

AsyncReqRes *CreateAsyncReqRes();
void DestroyAsyncReqRes(AsyncReqRes* req_res);
void PrintReqResCount();

class RequestHandle {
 public:
  RequestHandle();
  ~RequestHandle();
  void SetStarted();
  void SetCompleted();
  bool IsCompleted();
  void Wait();
  void Recycle();

  AsyncReqRes* request_;  // request always deleted
  AsyncReqRes* response_;
 private:
  bool started_;
  bool completed_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

class AsyncCommunicator {
 public:
  AsyncCommunicator() = delete;
  AsyncCommunicator(Partitioner *partitioner,
                    MemoryAllocatorBase *allocator,
                    RunnerRegistry *runner_registry,
                    Config *config);
  ~AsyncCommunicator();

  void CreateResources();
  void DestroyResources();
  void Start();

  Partitioner* GetPartitioner() {
    return partitioner_;
  }
  MemoryAllocatorBase* GetMemoryAllocator() {
    return allocator_;
  }

  // [INTERFACE]
  // request will be deleted after send, all memory_context will be free.
  void PutRequestAsync(RequestHandle* request_handle, bool need_notify_oneway_req = false);
  // [INTERFACE]
  // request will be deleted after send, all memory_context will be free.
  void PutRequestSync(AsyncReqRes* request, AsyncReqRes* response);
  // [INTERFACE]
  // response will be deleted after send, all memory_context will be free.
  // as response has no message back, so no sync interface and no handle.
  void PutResponseAsync(AsyncReqRes* response);
  // [INTERFACE]
  // send stop signal to remote rank, tell remote rank that this rank will have no further requests.
  void SendStopSignal();

  void WaitStopped();

  // for communicator to call
  void OnReceiveRequest(AsyncReqRes* request);
  // for communicator to call
  void OnReceiveResponse(AsyncReqRes* response);

  AsyncReqRes* OnRunnerGetResponse(AsyncReqRes* request);

  AsyncReqRes* OnReceiveGetResponse(Meta* meta);

  SideBandCommunicator* GetSideBandCommunicator() {
    return sideband_communicator_.get();
  }
 private:
  void SendStopSignalToSingleRank(int global_rank);

  Partitioner* partitioner_ = nullptr;
  MemoryAllocatorBase* allocator_ = nullptr;
  RunnerRegistry* runner_registry_ = nullptr;
  Config* config_ = nullptr;
  std::unique_ptr<MergedHandler> merged_handler_;

  MessageQueue<AsyncReqRes*> response_queue_;

  std::unique_ptr<IntraNodeCommunicator> intra_node_communicator_;
  std::unique_ptr<InterNodeCommunicator> inter_node_communicator_;

  std::mutex mutex_;
  std::unordered_map<uint64_t, RequestHandle*> pending_mapping_;

  std::unique_ptr<SideBandCommunicator> sideband_communicator_;

  friend class MergedHandler;
};