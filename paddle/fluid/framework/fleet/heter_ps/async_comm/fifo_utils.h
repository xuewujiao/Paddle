#pragma once

#include <string>

void SetFifoNamePrefix(const std::string& prefix);
std::string GetFifoNamePrefix();

// Read side should call CreateFifo
int CreateFifo(const std::string& fifo_name);
// Write side wait file created.
int WaitAndOpenFifo(const std::string& fifo_name);
// Unlink should be called on read side.
void UnlinkFifo(const std::string& fifo_name);

void SingleFifoWrite(int fifo_fd, const void* data, size_t data_size);
ssize_t SingleFifoRead(int fifo_fd, void* data, size_t data_size);
// returns true if time out
bool SingleFifoTimedRead(int fifo_fd, void* data, size_t data_size, ssize_t* read_size, int timeout_ms);
