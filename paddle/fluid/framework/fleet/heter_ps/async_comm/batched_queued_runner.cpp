#include "batched_queued_runner.h"

#include "async_communicator.h"
#include "check_macros.h"
#include "synchronizer.h"

int BatchedQueuedRunner::WorkItemList::WaitAndPopBatched(std::vector<QueuedRunner::RunnerWorkItem>* items,
                                                         const std::vector<FunctionInfo>& func_info) {
  items->clear();
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this](){ return (!request_list_.empty()) || finished_.load(); });
  if (finished_.load() && request_list_.empty()) return 0;
  items->push_back(std::move(request_list_.front()));
  request_list_.pop_front();
  int count = 1;
  int func_id = items->at(0).request->meta.function_id;
  BOOL_CHECK(func_id >= 0 && 0 < func_info.size() - func_id);
  
  auto it = request_list_.begin();
  int max_batchsize = func_info[func_id].max_batchsize;
  if (max_batchsize < 0) max_batchsize = INT32_MAX;
  while (it != request_list_.end() && count < max_batchsize) {
    if (it->request->meta.function_id == func_id) {
      items->push_back(*it);
      count++;
      it = request_list_.erase(it);
    } else {
      ++it;
    }
  }
  lock.unlock();
  finish_cv_.notify_one();
  return count;
}

void BatchedQueuedRunner::Process(AsyncReqRes *request,
                                  ProcessCallBackFuncType process_call_back_func) {
  if (IsStopMeta(&request->meta)) {
    if (MarkFinishedRank(request->meta.requester_node_id, request->meta.requester_lane_id)) {
      work_list_.Flush();
    }
    return;
  }
  RunnerWorkItem work_item;
  work_item.request = request;
  work_item.process_call_back_func = process_call_back_func;
  work_list_.Push(work_item);
}

void BatchedQueuedRunner::ProcessLoop() {
  ProcessSetup();
  std::vector<RunnerWorkItem> runner_work_items;
  while (work_list_.WaitAndPopBatched(&runner_work_items, function_info_table_) > 0) {
    BOOL_CHECK(!runner_work_items.empty());
    int func_id = runner_work_items[0].request->meta.function_id;
    BOOL_CHECK(func_id >= 0 && func_id < GetProcessFunctionCount());
    BOOL_CHECK(
        function_info_table_[func_id].batched_func_ != nullptr || function_info_table_[func_id].max_batchsize == 1);
    bool need_response = GetProcessFunctionInfoTable()[func_id].need_response;
    std::vector<AsyncReqRes*> requests(runner_work_items.size(), nullptr);
    std::vector<AsyncReqRes*> responses(runner_work_items.size(), nullptr);
    if (need_response) {
      for (size_t i = 0; i < runner_work_items.size(); i++) {
        requests[i] = runner_work_items[i].request;
        responses[i] = async_communicator_->OnRunnerGetResponse(runner_work_items[i].request);
      }
    }
    {
      SensitiveZoneGuard sensitive_zone_guard(partitioner_->GetLocalRank());
      if (requests.size() == 1) {
        function_info_table_[func_id].func_(requests[0], responses[0]);
      } else {
        BOOL_CHECK(function_info_table_[func_id].batched_func_ != nullptr);
        function_info_table_[func_id].batched_func_(requests.data(), responses.data(), requests.size());
      }
    }
    for (size_t i = 0; i < runner_work_items.size(); i++) {
      if (responses[i] != nullptr) {
        runner_work_items[i].process_call_back_func(responses[i]);
      }
      allocator_->FreeReqRes(runner_work_items[i].request);
      runner_work_items[i].Clear();
    }
  }
  work_queue_.Flush();
  ProcessCleanUp();
}
