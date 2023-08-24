#include "sideband_communicator.h"

#include <string.h>
#include <unistd.h>

#include <memory>
#include <thread>

#include "check_macros.h"
#include "log_macros.h"
#include "net_utils.h"

SideBandCommunicator::SideBandCommunicator(Partitioner *partitioner,
                                           const std::string &server_address,
                                           int server_port) :
    partitioner_(partitioner),
    server_address_(server_address),
    server_port_(server_port) {
  server_fds_.resize(partitioner->GetGlobalSize(), -1);
}

SideBandCommunicator::~SideBandCommunicator() {
}

void SideBandCommunicator::Start() {
  std::unique_ptr<std::thread> server_accept_thread;
  if (partitioner_->GetGlobalRank() == 0) {
    server_accept_thread = std::make_unique<std::thread>([this]() {
      this->ServerAcceptFunc();
    });
  }
  client_fd_ = CreateClientFd(server_address_, server_port_);
  int send_data[2];
  send_data[0] = kSideBandMagic;
  send_data[1] = partitioner_->GetGlobalRank();
  SingleSend(client_fd_, &send_data[0], sizeof(int) * 2);
  int magic_number = 0;
  SingleRecv(client_fd_, &magic_number, sizeof(int));
  BOOL_CHECK(magic_number == kSideBandMagic);
  if (partitioner_->GetGlobalRank() == 0) {
    server_accept_thread->join();
  }
  LOG_INFO("[Client] Rank=%d connected to server.", partitioner_->GetGlobalRank());
}

void SideBandCommunicator::Stop() {
  CALL_CHECK(close(client_fd_));
  client_fd_ = -1;
  if (partitioner_->GetGlobalRank() == 0) {
    for (int i = 0; i < partitioner_->GetGlobalSize(); i++) {
      CALL_CHECK(close(server_fds_[i]));
      server_fds_[i] = - 1;
    }
    server_fds_.clear();
  }
}

void SideBandCommunicator::ServerAcceptFunc() {
  int server_listen_fd = CreateServerListenFd(server_port_);
  // Listening
  ServerListen(server_listen_fd, partitioner_->GetGlobalSize());

  std::set<int> unconnected_rank_set;
  for (int i = 0; i < partitioner_->GetGlobalSize(); i++) {
    unconnected_rank_set.insert(i);
  }
  while (!unconnected_rank_set.empty()) {
    sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    int client_sock = accept(server_listen_fd, (sockaddr*)&client_addr, &client_addr_len);
    if (client_sock >= 0) {
      int recv_data[2];
      SingleRecv(client_sock, &recv_data[0], sizeof(int) * 2);
      BOOL_CHECK(recv_data[0] == kSideBandMagic);
      int rank_id = recv_data[1];
      BOOL_CHECK(rank_id >= 0 && rank_id < partitioner_->GetGlobalSize());
      BOOL_CHECK(unconnected_rank_set.count(rank_id) > 0);
      server_fds_[rank_id] = client_sock;
      unconnected_rank_set.erase(rank_id);
      LOG_INFO("[Server] Rank %d connected to SideBandCommunicator", rank_id);
    }
  }
  CALL_CHECK(close(server_listen_fd));
  LOG_INFO("[Server] All ranks connected to SideBandCommunicator");
  for (int i = 0; i < partitioner_->GetGlobalSize(); i++) {
    int recv_data[2];
    recv_data[0] = kSideBandMagic;
    recv_data[1] = i;
    SingleSend(server_fds_[i], &recv_data[0], sizeof(int));
  }
}

void SideBandCommunicator::RailAllToAll(const void *input,
                                        void *output,
                                        size_t element_size) {
  SingleSend(client_fd_, input, element_size * partitioner_->GetNodeCount());
  if (partitioner_->GetGlobalRank() == 0) {
    std::vector<char> recv_buffer(element_size * partitioner_->GetNodeCount());
    std::vector<std::vector<char>> send_buffer(partitioner_->GetNodeCount());
    for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
      send_buffer[i].resize(element_size * partitioner_->GetNodeCount());
    }
    for (int lr = 0; lr < partitioner_->GetRanksPerNode(); lr++) {
      for (int n = 0; n < partitioner_->GetNodeCount(); n++) {
        int r = partitioner_->MakeGlobalRank(n, lr);
        memset(recv_buffer.data(), 0, recv_buffer.size());
        SingleRecv(server_fds_[r], recv_buffer.data(), recv_buffer.size());
        for (int tn = 0; tn < partitioner_->GetNodeCount(); tn++) {
          memcpy(send_buffer[tn].data() + n * element_size, recv_buffer.data() + tn * element_size, element_size);
        }
      }
      for (int tn = 0; tn < partitioner_->GetNodeCount(); tn++) {
        int r = partitioner_->MakeGlobalRank(tn, lr);
        SingleSend(server_fds_[r], send_buffer[tn].data(), send_buffer[tn].size());
      }
    }
  }
  SingleRecv(client_fd_, output, element_size * partitioner_->GetNodeCount());
}

void SideBandCommunicator::RailAllGather(const void *input,
                                         void *output,
                                         size_t element_size) {
  SingleSend(client_fd_, input, element_size);
  if (partitioner_->GetGlobalRank() == 0) {
    std::vector<char> recv_buffer(element_size);
    std::vector<std::vector<char>> send_buffer(partitioner_->GetNodeCount());
    for (int i = 0; i < partitioner_->GetNodeCount(); i++) {
      send_buffer[i].resize(element_size * partitioner_->GetNodeCount());
    }
    for (int lr = 0; lr < partitioner_->GetRanksPerNode(); lr++) {
      for (int n = 0; n < partitioner_->GetNodeCount(); n++) {
        int r = partitioner_->MakeGlobalRank(n, lr);
        memset(recv_buffer.data(), 0, recv_buffer.size());
        SingleRecv(server_fds_[r], recv_buffer.data(), recv_buffer.size());
        for (int tn = 0; tn < partitioner_->GetNodeCount(); tn++) {
          memcpy(send_buffer[tn].data() + n * element_size, recv_buffer.data(), element_size);
        }
      }
      for (int tn = 0; tn < partitioner_->GetNodeCount(); tn++) {
        int r = partitioner_->MakeGlobalRank(tn, lr);
        SingleSend(server_fds_[r], send_buffer[tn].data(), send_buffer[tn].size());
      }
    }
  }
  SingleRecv(client_fd_, output, element_size * partitioner_->GetNodeCount());
}

void SideBandCommunicator::Barrier() {
  int rank = partitioner_->GetGlobalRank();
  SingleSend(client_fd_, &rank, sizeof(int));
  rank = -1;
  if (partitioner_->GetGlobalRank() == 0) {
    for (int r = 0; r < partitioner_->GetGlobalSize(); r++) {
      int rr;
      SingleRecv(server_fds_[r], &rr, sizeof(int));
      BOOL_CHECK(rr == r);
    }
    for (int r = 0; r < partitioner_->GetGlobalSize(); r++) {
      SingleSend(server_fds_[r], &r, sizeof(int));
    }
  }
  SingleRecv(client_fd_, &rank, sizeof(int));
  BOOL_CHECK(rank == partitioner_->GetGlobalRank());
}

