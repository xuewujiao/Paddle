#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "agent_copy_message.h"

class AsyncCommunicator;
class AsyncReqRes;
class Partitioner;
class MemoryAllocatorBase;

class CopyThread {
 public:
  CopyThread(Partitioner* partitioner, MemoryAllocatorBase* allocator);
  ~CopyThread();
  void CreateResources();
  void DestroyResources();
  void ConnectFifo();
  void SetAsyncCommunicator(AsyncCommunicator* async_communicator) {
    async_communicator_ = async_communicator;
  }
  void Start();
  void Stop();
  void WaitStopped();
  void Send(AsyncReqRes* req_res);
 private:
  void SendThreadFunc();
  void RecvThreadFunc();

  int recv_fifo_fd_ = -1;
  std::vector<int> send_fifo_fds_;
  int send_to_rail_fifo_fd_ = -1;

  std::condition_variable send_cv_;
  std::mutex send_mutex_;
  std::vector<std::queue<AsyncReqRes*>> send_queues_;
  std::vector<std::queue<IntraNodeCredit>> send_credits_;

  std::unique_ptr<std::thread> send_thread_;
  std::unique_ptr<std::thread> recv_thread_;

  Partitioner* partitioner_ = nullptr;
  MemoryAllocatorBase* allocator_ = nullptr;
  AsyncCommunicator* async_communicator_ = nullptr;

  std::atomic<bool> stopped_{};
  std::atomic<bool> sender_exited_{};
  std::atomic<int> total_credit_count_{};
};