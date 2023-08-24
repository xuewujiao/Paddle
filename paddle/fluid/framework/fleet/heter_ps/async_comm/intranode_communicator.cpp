#include "intranode_communicator.h"

#include <string.h>

#include "async_communicator.h"
#include "check_macros.h"
#include "cuda_utils.h"
#include "fifo_utils.h"
#include "log_macros.h"
#include "meta.h"

IntraNodeCommunicator::IntraNodeCommunicator(Partitioner *partitioner, MemoryAllocatorBase *allocator) :
    partitioner_(partitioner), allocator_base_(allocator) {
  stopped_.store(false);
  need_del_req_res_.store(0);
}

IntraNodeCommunicator::~IntraNodeCommunicator() {

}

void IntraNodeCommunicator::CreateResources() {
  req_read_fifo_ =
      CreateFifo(GetFifoNameForIntraNodeCommunicator("req", partitioner_->GetNodeID(), partitioner_->GetLocalRank()));
  res_read_fifo_ =
      CreateFifo(GetFifoNameForIntraNodeCommunicator("res", partitioner_->GetNodeID(), partitioner_->GetLocalRank()));
}

void IntraNodeCommunicator::DestroyResources() {
  UnlinkFifo(GetFifoNameForIntraNodeCommunicator("req", partitioner_->GetNodeID(), partitioner_->GetLocalRank()));
  UnlinkFifo(GetFifoNameForIntraNodeCommunicator("res", partitioner_->GetNodeID(), partitioner_->GetLocalRank()));
}

void IntraNodeCommunicator::ConnectFifos() {
  req_write_fifo_.resize(partitioner_->GetRanksPerNode(), -1);
  res_write_fifo_.resize(partitioner_->GetRanksPerNode(), -1);
  for (int lr = 0; lr < partitioner_->GetRanksPerNode(); lr++) {
    req_write_fifo_[lr] = WaitAndOpenFifo(GetFifoNameForIntraNodeCommunicator("req", partitioner_->GetNodeID(), lr));
    res_write_fifo_[lr] = WaitAndOpenFifo(GetFifoNameForIntraNodeCommunicator("res", partitioner_->GetNodeID(), lr));
  }
}

std::string IntraNodeCommunicator::GetFifoNameForIntraNodeCommunicator(const std::string &type,
                                                                       int node_id,
                                                                       int local_rank) {
  std::string name = GetFifoNamePrefix();
  name.append("intranode_node_").append(type).append("_").append(std::to_string(node_id))
      .append("_lr_").append(std::to_string(local_rank));
  return name;
}

enum IntraNodeFifoMsgType {
  INFMT_NONE = 0,
  INFMT_REQRES = 1,
  INFMT_DEL = 2,
};

struct InterNodeFifoMsg {
  IntraNodeFifoMsgType msg_type = INFMT_NONE;
  AsyncReqRes *req_res = nullptr;
};

void IntraNodeCommunicator::Send(AsyncReqRes *req_res) {
  BOOL_CHECK(IsLocalNode(&req_res->meta));
  BOOL_CHECK(req_res->meta.is_response == 0 || req_res->meta.is_response == 1);
  if (!IsLocalGPU(&req_res->meta)) {
    need_del_req_res_.fetch_add(1);
  }
  int target_local_rank =
      req_res->meta.is_response == 1 ? req_res->meta.requester_lane_id : req_res->meta.runner_lane_id;
  InterNodeFifoMsg msg{};
  msg.msg_type = INFMT_REQRES;
  msg.req_res = req_res;
  if (req_res->meta.is_response == 0) {
    SingleFifoWrite(req_write_fifo_[target_local_rank], &msg, sizeof(InterNodeFifoMsg));
  } else {
    SingleFifoWrite(res_write_fifo_[target_local_rank], &msg, sizeof(InterNodeFifoMsg));
  }
}

void IntraNodeCommunicator::Start() {
  req_thread_ = std::make_unique<std::thread>([this]() {
    this->RequestProcessLoop();
  });
  res_thread_ = std::make_unique<std::thread>([this]() {
    this->ResponseProcessLoop();
  });
}

void IntraNodeCommunicator::Stop() {
  stopped_.store(true);
  while (need_del_req_res_.load() > 0) {
    usleep(1000);
  }
  LOG_INFO("Rank=%d, shutting down IntraNodeCommunicator", partitioner_->GetGlobalRank());
  for (int lr = 0; lr < partitioner_->GetRanksPerNode(); lr++) {
    CALL_CHECK(close(req_write_fifo_[lr]));
    CALL_CHECK(close(res_write_fifo_[lr]));
  }
  LOG_INFO("Rank=%d, IntraNodeCommunicator shutdown", partitioner_->GetGlobalRank());
}

void IntraNodeCommunicator::WaitStopped() {
  req_thread_->join();
  res_thread_->join();
  LOG_INFO("IntraNodeCommunicator Stopped for rank=%d", partitioner_->GetGlobalRank());
}

static void CopyRequestResponseLocalNode(AsyncReqRes *dst,
                                         const AsyncReqRes *src,
                                         MemoryAllocatorBase *allocator,
                                         cudaStream_t stream) {
  dst->meta = src->meta;
  allocator->AllocateReqResByMeta(dst);
  for (int i = 0; i < dst->meta.valid_data_count; i++) {
    MemcpyUnique(dst->memory_contexts[i]->GetPointer(),
                 src->memory_contexts[i]->GetPointer(),
                 dst->meta.data_sizes[i],
                 stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void IntraNodeCommunicator::RequestProcessLoop() {
  CUDA_CHECK(cudaSetDevice(partitioner_->GetLocalRank()));
  cudaStream_t req_stream = CreateHighPriorityStream(cudaStreamNonBlocking);
  while (!stopped_.load() || need_del_req_res_.load() > 0) {
    InterNodeFifoMsg msg{};
    auto bytes = SingleFifoRead(req_read_fifo_, &msg, sizeof(InterNodeFifoMsg));
    if (bytes == 0) {
      break;
    }
    BOOL_CHECK(bytes == sizeof(InterNodeFifoMsg));
    BOOL_CHECK(msg.msg_type == INFMT_REQRES || msg.msg_type == INFMT_DEL);
    BOOL_CHECK(msg.req_res->meta.is_response == 0);
    if (msg.msg_type == INFMT_REQRES) {
      if (IsLocalGPU(&msg.req_res->meta)) {
        async_communicator_->OnReceiveRequest(msg.req_res);
      } else {
        auto *request = CreateAsyncReqRes();
        CopyRequestResponseLocalNode(request, msg.req_res, allocator_base_, req_stream);
        async_communicator_->OnReceiveRequest(request);
        int from_local_rank = msg.req_res->meta.requester_lane_id;
        msg.msg_type = INFMT_DEL;
        SingleFifoWrite(req_write_fifo_[from_local_rank], &msg, sizeof(InterNodeFifoMsg));
      }
    } else {
      allocator_base_->FreeReqRes(msg.req_res);
      need_del_req_res_.fetch_add(-1);
    }
  }
  CALL_CHECK(close(req_read_fifo_));
  CUDA_CHECK(cudaStreamDestroy(req_stream));
}

void IntraNodeCommunicator::ResponseProcessLoop() {
  CUDA_CHECK(cudaSetDevice(partitioner_->GetLocalRank()));
  cudaStream_t res_stream = CreateHighPriorityStream(cudaStreamNonBlocking);
  while (!stopped_.load() || need_del_req_res_.load() > 0) {
    InterNodeFifoMsg msg{};
    auto bytes = SingleFifoRead(res_read_fifo_, &msg, sizeof(InterNodeFifoMsg));
    if (bytes == 0) {
      break;
    }
    BOOL_CHECK(bytes == sizeof(InterNodeFifoMsg));
    BOOL_CHECK(msg.msg_type == INFMT_REQRES || msg.msg_type == INFMT_DEL);
    BOOL_CHECK(msg.req_res->meta.is_response == 1);
    if (msg.msg_type == INFMT_REQRES) {
      if (IsLocalGPU(&msg.req_res->meta)) {
        async_communicator_->OnReceiveResponse(msg.req_res);
      } else {
        auto *response = async_communicator_->OnReceiveGetResponse(&msg.req_res->meta);
        CopyRequestResponseLocalNode(response, msg.req_res, allocator_base_, res_stream);
        async_communicator_->OnReceiveResponse(response);
        int from_local_rank = msg.req_res->meta.runner_lane_id;
        msg.msg_type = INFMT_DEL;
        SingleFifoWrite(res_write_fifo_[from_local_rank], &msg, sizeof(InterNodeFifoMsg));
      }
    } else {
      allocator_base_->FreeReqRes(msg.req_res);
      need_del_req_res_.fetch_add(-1);
    }
  }
  CALL_CHECK(close(res_read_fifo_));
  CUDA_CHECK(cudaStreamDestroy(res_stream));
}
