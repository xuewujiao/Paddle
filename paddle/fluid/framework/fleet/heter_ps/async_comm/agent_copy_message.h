#pragma once

#include <string>

#include "meta.h"

// Copy and Agent messages
// Send path:
// 1. Agent->Copy: Add Buffer Credit.
// 2. Copy->Agent: Buffer Filled (if it is stop signal).
// Recv path:
// 3. Agent->Copy: Buffer Ready. (if it is stop signal).
// 4. Copy->Agent: Buffer Consumed.

struct IntraNodeCredit {
  void *pointer;
};

struct Header {
  Meta meta;
};

constexpr size_t kRegBufferSize = 1 << 22;
constexpr int kRegBufferCount = 4;
constexpr size_t kDataSizePerBuffer = kRegBufferSize - sizeof(Header);

struct AgentCopyMessage {
  AgentCopyMessage() {
    buffer_ptr = nullptr;
    data_size = 0;
    // src and dst are data flow src and dst, credit is reversed
    src_global_rank = -1;
    dst_global_rank = -1;
    has_data = 0;
    is_stop_signal = 0;
  }
  void MakeMsg(void* ptr, size_t content_data_size, int src_gr, int dst_gr, int with_data, bool stop_signal) {
    buffer_ptr = ptr;
    data_size = content_data_size;
    src_global_rank = src_gr;
    dst_global_rank = dst_gr;
    has_data = with_data;
    is_stop_signal = stop_signal ? 1 : 0;
  }
  void* buffer_ptr = nullptr;
  size_t data_size;
  int src_global_rank : 15;
  int dst_global_rank : 15;
  uint64_t has_data : 1;
  uint64_t is_stop_signal : 1;
};

std::string GetAgentCopyFifoName(const std::string& type, int node_id, int local_rank);

