#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <list>
#include <mutex>
#include <vector>

#include "runner.h"

class BatchedQueuedRunner : public QueuedRunner {
 public:
  BatchedQueuedRunner() = delete;
  explicit BatchedQueuedRunner(Partitioner* partitioner, MemoryAllocatorBase* allocator)
      : QueuedRunner(partitioner, allocator) {}
  ~BatchedQueuedRunner() override = default;
  void Process(AsyncReqRes* request, ProcessCallBackFuncType process_call_back_func) override;
 protected:
  class WorkItemList {
   public:
    WorkItemList() {
      finished_.store(false);
    }
    ~WorkItemList() {
      WaitUntilFinished();
    }
    void Push(const RunnerWorkItem& item) {
      std::unique_lock<std::mutex> lock(mu_);
      // should not Push new items after Flush.
      assert(finished_.load() == false);
      request_list_.push_back(item);
      lock.unlock();
      cv_.notify_one();
    }
    bool WaitAndPop(RunnerWorkItem* item) {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this](){ return (!request_list_.empty()) || finished_.load(); });
      if (finished_.load() && request_list_.empty()) return false;
      *item = std::move(request_list_.front());
      request_list_.pop_front();
      lock.unlock();
      finish_cv_.notify_one();
      return true;
    }
    int WaitAndPopBatched(std::vector<RunnerWorkItem> *items,
                          const std::vector<FunctionInfo> &batched_func_info);
    bool Finished() const {
      std::lock_guard<std::mutex> lock(mu_);
      return finished_;
    }
    void Flush() {
      std::unique_lock<std::mutex> lock(mu_);
      finished_ = true;
      lock.unlock();
      cv_.notify_all();
      finish_cv_.notify_one();
    }
    bool Empty() const {
      std::lock_guard<std::mutex> lock(mu_);
      return request_list_.empty();
    }
    size_t Size() const {
      std::lock_guard<std::mutex> lock(mu_);
      return request_list_.size();
    }
    void WaitUntilFinished() {
      std::unique_lock<std::mutex> lock(mu_);
      finish_cv_.wait(lock, [this](){ return request_list_.empty() && finished_.load(); });
    }
   private:
    std::atomic<bool> finished_{};
    std::list<RunnerWorkItem> request_list_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable finish_cv_;
  };
  virtual void ProcessLoop() override;
  WorkItemList work_list_;
};


