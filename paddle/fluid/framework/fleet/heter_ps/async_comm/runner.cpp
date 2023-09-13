#include "runner.h"

#include <memory>

#include "async_communicator.h"
#include "check_macros.h"
#include "log_macros.h"

static constexpr int kMaxRankCountBits = 12;
static constexpr int kRankIdShift = 64 - kMaxRankCountBits;
static constexpr int kMaxRunnerCountBits = 4;
static constexpr int kRunnerIdShift = kRankIdShift - kMaxRunnerCountBits;

void RunnerBase::Register(int runner_id) {
  runner_id_ = runner_id;

  uint64_t global_rank_offset = partitioner_->GetGlobalRank();
  global_rank_offset <<= kRankIdShift;
  uint64_t runner_id_offset = runner_id_;
  runner_id_offset <<= kRunnerIdShift;
  request_id_.store(global_rank_offset | runner_id_offset);

  RegisterFunctions();
}

int RunnerBase::GetProcessFunctionCount() const {
  return static_cast<int>(function_info_table_.size());
}

const FunctionInfo *RunnerBase::GetProcessFunctionInfoTable() const {
  return function_info_table_.data();
}

bool RunnerBase::FuncNeedResponse(Meta *meta) const {
  // don't need response for stop signal
  if (meta->status_code == -1) return false;
  return function_info_table_[meta->function_id].need_response;
}

void RunnerBase::CreateRequestMeta(Meta *meta, int target_global_rank, int function_id) {
  int target_node_id = partitioner_->GetNodeIDFromGlobalRank(target_global_rank);
  int target_local_rank = partitioner_->GetLocalrankFromGlobalRank(target_global_rank);
  meta->status_code = function_id == -1 ? -1 : 0;
  meta->runner_id = static_cast<int16_t>(runner_id_);
  meta->function_id = static_cast<int16_t>(function_id);
  meta->requester_node_id = static_cast<int16_t>(partitioner_->GetNodeID());
  meta->requester_lane_id = static_cast<int8_t>(partitioner_->GetLocalRank());
  meta->runner_node_id = static_cast<int16_t>(target_node_id);
  meta->runner_lane_id = static_cast<int8_t>(target_local_rank);
  meta->request_id = GenerateRequestID();
}

uint64_t RunnerBase::GenerateRequestID() {
  return request_id_.fetch_add(1);
}

bool QueuedRunner::MarkFinishedRank(int node_id, int local_rank) {
  std::unique_lock<std::mutex> mlock(mutex_);
  active_rank_set_->RemoveRank(node_id, local_rank);
  return active_rank_set_->Empty();
}

void QueuedRunner::Process(AsyncReqRes *request,
                           ProcessCallBackFuncType process_call_back_func) {
  if (IsStopMeta(&request->meta)) {
    if (MarkFinishedRank(request->meta.requester_node_id, request->meta.requester_lane_id)) {
      work_queue_.Flush();
    }
    return;
  }
  RunnerWorkItem work_item;
  work_item.request = request;
  work_item.process_call_back_func = process_call_back_func;
  work_queue_.Push(work_item);
}

void QueuedRunner::ProcessLoop() {
  ProcessSetup();
  RunnerWorkItem runner_work_item;
  while (work_queue_.WaitAndPop(&runner_work_item)) {
	//VLOG(0) << get_runner_name() << " queue size is " << work_queue_.Size();
    int func_id = runner_work_item.request->meta.function_id;
    BOOL_CHECK(func_id >= 0 && func_id < GetProcessFunctionCount());
    AsyncReqRes *response = nullptr;
    if (GetProcessFunctionInfoTable()[func_id].need_response) {
      response = async_communicator_->OnRunnerGetResponse(runner_work_item.request);
    }
    GetProcessFunctionInfoTable()[func_id].func_(runner_work_item.request, response);
    if (response != nullptr) {
      runner_work_item.process_call_back_func(response);
    }
    allocator_->FreeReqRes(runner_work_item.request);
    runner_work_item.Clear();
  }
  ProcessCleanUp();
}

void QueuedRunner::StartProcessLoop() {
  active_rank_set_ = std::make_unique<RankSet>(partitioner_->GetNodeCount(), partitioner_->GetRanksPerNode());
  active_rank_set_->MakeFullSet();
  process_thread_ = std::make_unique<std::thread>([this]() {
    this->ProcessLoop();
  });
}

void QueuedRunner::WaitStopped() {
  process_thread_->join();
}

void RunnerRegistry::Register(int runner_id, RunnerBase *runner) {
  auto it = registry_.find(runner_id);
  BOOL_CHECK(it == registry_.end());
  runner->Register(runner_id);
  registry_.insert(std::make_pair(runner_id, runner));
}

RunnerBase *RunnerRegistry::Find(int runner_id) {
  auto it = registry_.find(runner_id);
  BOOL_CHECK(it != registry_.end());
  return it->second;
}

void RunnerRegistry::Traverse(const std::function<void(RunnerBase *)> &fn) {
  auto it = registry_.begin();
  for (; it != registry_.end(); ++it) {
    fn(it->second);
  }
}
