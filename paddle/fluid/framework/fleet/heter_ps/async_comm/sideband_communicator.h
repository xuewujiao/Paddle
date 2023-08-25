#pragma once

#include <string>
#include <vector>

#include "partitioner.h"

class SideBandCommunicator {
 public:
  SideBandCommunicator(Partitioner* partitioner, const std::string& server_address, int server_port);
  ~SideBandCommunicator();
  void Start();
  void Stop();
  void RailAllToAll(const void* input, void* output, size_t element_size);
  void RailAllGather(const void* input, void* output, size_t element_size);
  void Barrier();
 protected:
  static constexpr int kSideBandMagic = 0x51debacd;
  void ServerAcceptFunc();
  int client_fd_ = -1;
  std::vector<int> server_fds_;
  Partitioner* partitioner_ = nullptr;
  std::string server_address_;
  int server_port_ = -1;
};