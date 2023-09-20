#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "memory_allocator.h"
#include "message_queue.h"
#include "meta.h"
#include "partitioner.h"
#include "glog/logging.h"

class AsyncReqRes;
class AsyncCommunicator;
typedef std::function<void(AsyncReqRes*, AsyncReqRes*)> ProcessFuncType;
typedef std::function<void(AsyncReqRes*)> ProcessCallBackFuncType;
typedef std::function<void(AsyncReqRes**, AsyncReqRes**, size_t)> BatchedProcessFuncType;

struct FunctionInfo {
  int16_t function_id = -1;
  int16_t input_data_count = 0;
  int16_t output_data_count = 0;
  bool need_response = false;
  ProcessFuncType func_ = nullptr;
  int max_batchsize = 1;
  BatchedProcessFuncType batched_func_ = nullptr;
  int8_t input_locations[MAX_VEC_COUNT] = {};
  int8_t output_locations[MAX_VEC_COUNT] = {};
};

class RunnerBase {
 public:
  RunnerBase() = delete;
  explicit RunnerBase(Partitioner* partitioner, MemoryAllocatorBase* allocator)
      : partitioner_(partitioner), allocator_(allocator) {}
  virtual ~RunnerBase() {};
  void Register(int runner_id);
  int RunnerId() const {
    return runner_id_;
  }
  // Normally GetProcessFunctionCount and GetProcessFunctionInfoTable don't need to be overwritten.
  virtual int GetProcessFunctionCount() const;
  virtual std::string get_runner_name() {return std::string("RunnerBase");}
  virtual const FunctionInfo* GetProcessFunctionInfoTable() const;
  bool FuncNeedResponse(Meta* meta) const;
  // [INTERFACE]
  // see QueuedRunner implementation
  virtual void Process(AsyncReqRes *request, ProcessCallBackFuncType process_call_back_func) = 0;
  void CreateRequestMeta(Meta* meta, int target_global_rank, int function_id);
  virtual void WaitStopped() = 0;
  void SetAsyncCommunicator(AsyncCommunicator* async_communicator) {
    async_communicator_ = async_communicator;
  }
 protected:
  // [INTERFACE]
  // In RegisterFunctions, function_info_table_ should be filled correctly.
  // These functions are processing functions
  virtual void RegisterFunctions() = 0;
  uint64_t GenerateRequestID();
  std::vector<FunctionInfo> function_info_table_;
  Partitioner* partitioner_ = nullptr;
  MemoryAllocatorBase* allocator_ = nullptr;
  int runner_id_ = -1;
  std::atomic<uint64_t> request_id_;
  AsyncCommunicator* async_communicator_ = nullptr;
};

class QueuedRunner : public RunnerBase {
 public:
  QueuedRunner() = delete;
  explicit QueuedRunner(Partitioner* partitioner, MemoryAllocatorBase* allocator)
      : RunnerBase(partitioner, allocator) {}
  ~QueuedRunner() {};
  virtual void StartProcessLoop();
  void WaitStopped() override;
  void Process(AsyncReqRes* request, ProcessCallBackFuncType process_call_back_func) override;
  virtual std::string get_runner_name() {return std::string("QueuedRunner");}
 protected:
  bool MarkFinishedRank(int node_id, int local_rank);
  // [INTERFACE] maybe set up cudaStream and set device.
  virtual void ProcessSetup() {}
  virtual void ProcessLoop();
  // [INTERFACE] maybe destroy cudaStream
  virtual void ProcessCleanUp() {}

  struct RunnerWorkItem {
    AsyncReqRes* request = nullptr;
    ProcessCallBackFuncType process_call_back_func = nullptr;
    void Clear() {
      request = nullptr;
      process_call_back_func = nullptr;
    }
  };
  MessageQueue<RunnerWorkItem> work_queue_;
  std::unique_ptr<std::thread> process_thread_;
  std::mutex mutex_;
  std::unique_ptr<RankSet> active_rank_set_;
};

class RunnerRegistry {
 public:
  RunnerRegistry() = default;
  ~RunnerRegistry() = default;
  void Register(int runner_id, RunnerBase* runner);
  RunnerBase* Find(int runner_id);
  void Traverse(const std::function<void(RunnerBase*)>& fn);
 private:
  std::unordered_map<int, RunnerBase*> registry_;
};
