#pragma once

#include <atomic>
#include <cassert>
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename Request>
class MessageQueue {
 public:
  MessageQueue() {
    finished_.store(false);
  }
  ~MessageQueue() {
    WaitUntilFinished();
  }
  void Push(const Request& item) {
    std::unique_lock<std::mutex> lock(mu_);
    // should not Push new items after Flush.
    assert(finished_.load() == false);
    request_queue_.push(item);
    lock.unlock();
    cv_.notify_one();
  }
  bool TryPop(Request* item) {
    std::unique_lock<std::mutex> lock(mu_);
    if (request_queue_.empty()) {
      return false;
    }
    *item = std::move(request_queue_.front());
    request_queue_.pop();
    lock.unlock();
    finish_cv_.notify_one();
    return true;
  }
  bool WaitAndPop(Request* item) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this](){ return (!request_queue_.empty()) || finished_.load(); });
    if (finished_.load() && request_queue_.empty()) return false;
    *item = std::move(request_queue_.front());
    request_queue_.pop();
    lock.unlock();
    finish_cv_.notify_one();
    return true;
  }
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
    return request_queue_.empty();
  }
  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return request_queue_.size();
  }
  void WaitUntilFinished() {
    std::unique_lock<std::mutex> lock(mu_);
    finish_cv_.wait(lock, [this](){ return request_queue_.empty() && finished_.load(); });
  }
 private:
  std::atomic<bool> finished_;
  std::queue<Request> request_queue_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable finish_cv_;
};
