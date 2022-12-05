// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <pthread.h>
#include <semaphore.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
class Barrier {
 public:
  explicit Barrier(int count = 1) {
    CHECK_GE(count, 1);
    CHECK_EQ(0, pthread_barrier_init(&_barrier, NULL, count));
  }
  ~Barrier() { CHECK_EQ(0, pthread_barrier_destroy(&_barrier)); }
  void reset(int count) {
    CHECK_GE(count, 1);
    CHECK_EQ(0, pthread_barrier_destroy(&_barrier));
    CHECK_EQ(0, pthread_barrier_init(&_barrier, NULL, count));
  }
  void wait() {
    int err = pthread_barrier_wait(&_barrier);
    CHECK_EQ(true,
             (err = pthread_barrier_wait(&_barrier),
              err == 0 || err == PTHREAD_BARRIER_SERIAL_THREAD));
  }

 private:
  pthread_barrier_t _barrier;
};
// Call func(args...). If interrupted by signal, recall the function.
template <class FUNC, class... ARGS>
auto ignore_signal_call(FUNC &&func, ARGS &&...args) ->
    typename std::result_of<FUNC(ARGS...)>::type {
  for (;;) {
    auto err = func(args...);

    if (err < 0 && errno == EINTR) {
      LOG(INFO) << "Signal is caught. Ignored.";
      continue;
    }
    return err;
  }
}
class Semaphore {
 public:
  Semaphore() { CHECK_EQ(0, sem_init(&_sem, 0, 0)); }
  ~Semaphore() { CHECK_EQ(0, sem_destroy(&_sem)); }
  void post() { CHECK_EQ(0, sem_post(&_sem)); }
  void wait() { CHECK_EQ(0, ignore_signal_call(sem_wait, &_sem)); }
  bool try_wait() {
    int err = 0;
    CHECK_EQ(true,
             (err = ignore_signal_call(sem_trywait, &_sem),
              err == 0 || errno == EAGAIN));
    return err == 0;
  }

 private:
  sem_t _sem;
};
class WaitGroup {
 public:
  WaitGroup() {}
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    counter_ = 0;
    cond_.notify_all();
  }
  void add(int delta) {
    if (delta == 0) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    counter_ += delta;
    if (counter_ == 0) {
      cond_.notify_all();
    }
  }
  void done() { add(-1); }
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);

    while (counter_ != 0) {
      cond_.wait(lock);
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cond_;
  int counter_ = 0;
};
}  // namespace framework
}  // namespace paddle
