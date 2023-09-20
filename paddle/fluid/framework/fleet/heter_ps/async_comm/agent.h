#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include "agent_copy_message.h"
#include "config.h"
#include "ib_utils.h"

class Partitioner;
class SideBandCommunicator;

struct InterNodeCredit {
  int target_node_id = -1;
  int buffer_idx = -1;
};

class Agent {
 public:
  explicit Agent(Partitioner* partitioner, Config* config);
  ~Agent();
  void SetSideBandCommunicator(SideBandCommunicator* side_band_communicator) {
    side_band_communicator_ = side_band_communicator;
  }
  void CreateResources();
  void DestroyResources();
  void ConnectFifo();
  void ConnectNetwork();
  void Start();
  void Stop();
  void WaitStopped();
 private:
  void ReadFifoThreadFunc();
  void PollCQThreadFunc();
  void SendToRemoteThreadFunc();

  int recv_fifo_fd_ = -1;
  std::vector<int> send_fifo_fds_;
  int send_to_rail_fifo_fd_ = -1;

  std::condition_variable send_cv_;
  std::mutex send_mutex_;

  union ImmData {
    uint32_t imm_u32;
    struct {
      int is_stop_signal : 1;
      int has_data : 1;
      // src and dst local rank are data flow src and dst, credit is reversed
      int src_local_rank : 4;
      int dst_local_rank : 4;
      int buffer_idx : 16;
    };
  };

  static_assert(sizeof(ImmData) == sizeof(uint32_t), "ImmData size incorrect.");

  union WrId {
    uint64_t id;
    struct {
      int send_buffer_idx;
      ImmData imm_data;
    };
  };
  static_assert(sizeof(WrId) == sizeof(uint64_t), "WrId size incorrect.");

  struct InterNodeToSendData {
    size_t data_size = 0;
    char* data_pointer = nullptr;
    ImmData imm_data;
  };

  std::vector<std::queue<InterNodeToSendData>> to_send_credits_;
  std::vector<std::queue<InterNodeToSendData>> internode_to_send_data_;
  std::vector<std::queue<InterNodeCredit>> internode_remote_recv_credits_;

  void AddInterNodeSendData(AgentCopyMessage agent_copy_message);
  int GetBufferIndex(char* ptr, bool is_recv);
  int GetNodeIdFromQpn(uint32_t qpn, bool is_recv);

  std::unique_ptr<std::thread> read_fifo_thread_;  // read recv_fifo_fd_, process buffer filled and buffer consumed msg.
  std::unique_ptr<std::thread> poll_cq_thread_;  // poll complete queue, send add buffer credit and buffer ready msg.
  std::unique_ptr<std::thread> rdma_send_thread_; // post rdma send request.

  Partitioner* partitioner_ = nullptr;
  Config* config_ = nullptr;  
  SideBandCommunicator* side_band_communicator_ = nullptr;

  IbLocalContext ib_local_context_{};
  std::vector<ibv_qp*> send_qps_;
  std::vector<ibv_qp*> recv_qps_;
  std::unordered_map<uint32_t, int> send_qpn_to_node_id_;
  std::unordered_map<uint32_t, int> recv_qpn_to_node_id_;
  char* send_reg_mem_ = nullptr;
  char* recv_reg_mem_ = nullptr;
  static constexpr size_t kCreditRegMemSize = 8;
  char* credit_reg_mem_ = nullptr;
  int64_t* imm_mem_ = nullptr;
  ibv_mr* send_mr_ = nullptr;
  ibv_mr* recv_mr_ = nullptr;
  ibv_mr* credit_mr_ = nullptr;
  ibv_mr* imm_mr_ = nullptr;

  struct RemoteNodeInfo {
    IbPeerInfo peer_recv_info;
    IbPeerInfo peer_send_info;
    IbMemInfo recv_mem_info{};
    IbMemInfo credit_mem_info{};
  };
  std::vector<RemoteNodeInfo> local_node_infos_;
  std::vector<RemoteNodeInfo> remote_node_infos_;

  std::atomic<bool> no_more_sends_{};
  std::atomic<int> pending_sends_{};
  std::atomic<bool> sender_exited_{};
  std::atomic<int> uncompleted_data_sends_{};
  std::atomic<int> total_credit_count_{};

  bool use_gpu_direct_rdma_ = true;
};