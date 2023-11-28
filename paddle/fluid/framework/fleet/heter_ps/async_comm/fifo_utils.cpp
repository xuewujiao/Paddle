#include "fifo_utils.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "log_macros.h"

static std::string s_fifo_name_prefix = "./async_copy_fifo_";

void SetFifoNamePrefix(const std::string& prefix) {
  s_fifo_name_prefix = prefix;
}

std::string GetFifoNamePrefix() {
  return s_fifo_name_prefix;
}

int CreateFifo(const std::string& fifo_name) {
  struct stat st;
  memset(&st, 0, sizeof(struct stat));
  if (stat(fifo_name.c_str(), &st) == 0) {
    LOG_FATAL("fifo %s already exist, use SetFifoNamePrefix to set a new prefix or you may need to remove that.",
              fifo_name.c_str());
    return -1;
  }

  if (mkfifo(fifo_name.c_str(), 0666) != 0) {
    LOG_FATAL("mkfifo failed file %s, errno=%d, %s", fifo_name.c_str(), errno, strerror(errno));
    return -1;
  }

  int fd = open(fifo_name.c_str(), O_RDONLY | O_NONBLOCK);
  if (fd < 0) {
    LOG_FATAL("open fifo file %s failed. errno=%d, %s", fifo_name.c_str(), errno, strerror(errno));
    return -1;
  }
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags == -1) {
    LOG_FATAL("fifo fcntl F_GETFL failed.");
  }
  flags &= ~O_NONBLOCK;
  if (fcntl(fd, F_SETFL, flags) == -1) {
    LOG_FATAL("fifo fcntl F_SETFL failed.");
  }

  return fd;
}

int WaitAndOpenFifo(const std::string& fifo_name) {
  struct stat st;
  memset(&st, 0, sizeof(struct stat));
  int time_since_last_print = 0;
  while(true) {
    // Check if fifo is already exist.
    if (stat(fifo_name.c_str(), &st) == 0) {
      // Check if it is fifo
      if (S_ISFIFO(st.st_mode)) {
        // open fifo and return fd
        int fd = open(fifo_name.c_str(), O_WRONLY);
        if (fd < 0) {
          LOG_FATAL("fifo file %s exist but open failed.", fifo_name.c_str());
          return -1;
        }
        return fd;
      } else {
        LOG_FATAL("file %s is not fifo", fifo_name.c_str());
        return -1;
      }
    }
    constexpr int wait_time_us = 10 * 1000;
    usleep(wait_time_us);  // poll
    time_since_last_print += wait_time_us;
    if (time_since_last_print >= 1000 * 1000) {
      LOG_INFO("Waiting fifo file %s", fifo_name.c_str());
      time_since_last_print = 0;
    }
  }
  return -1;
}

void UnlinkFifo(const std::string& fifo_name) {
  struct stat st;
  memset(&st, 0, sizeof(struct stat));
  if (stat(fifo_name.c_str(), &st) == 0) {
    if (unlink(fifo_name.c_str()) == -1) {
      LOG_FATAL("unlink fifo %s failed.", fifo_name.c_str());
    }
  } else {
    LOG_FATAL("cannot unlink fifo %s, file not exist", fifo_name.c_str());
  }
}

void SingleFifoWrite(int fifo_fd, const void* data, size_t data_size) {
  auto ret = write(fifo_fd, data, data_size);
  if ((size_t)ret != data_size) {
    LOG_FATAL("SingleFifoWrite %ld bytes to fifo failed. ret=%ld", data_size, ret);
  }
}

ssize_t SingleFifoRead(int fifo_fd, void* data, size_t data_size) {
  auto ret = read(fifo_fd, data, data_size);
  if (ret < 0) {
    LOG_FATAL("SingleFifoRead %ld bytes from fifo failed. ret=%ld", data_size, ret);
  }
  return ret;
}

bool SingleFifoTimedRead(int fifo_fd, void* data, size_t data_size, ssize_t* read_size, int timeout_ms) {
  struct pollfd pfd{};
  pfd.fd = fifo_fd;
  pfd.events = POLLIN;

  *read_size = 0;
  int ret = poll(&pfd, 1, timeout_ms);
  if (ret > 0) {
    *read_size = read(fifo_fd, data, data_size);
    if (*read_size < 0) {
      LOG_FATAL("SingleFifoTimedRead %ld bytes from fifo failed. ret=%ld", data_size, *read_size);
    }
    return false;
  } else if (ret == 0) {
    // timeout
    return true;
  } else {
    // error
    LOG_FATAL("SingleFifoTimedRead poll failed. ret=%d", ret);
    return false;
  }
}