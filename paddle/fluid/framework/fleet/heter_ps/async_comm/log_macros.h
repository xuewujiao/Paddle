#pragma once

#include <cassert>
#include <cstdarg>
#include <iostream>
#include <string>
#include <vector>

static constexpr int LEVEL_FATAL = 0;
static constexpr int LEVEL_ERROR = 10;
static constexpr int LEVEL_WARN  = 100;
static constexpr int LEVEL_INFO  = 1000;
static constexpr int LEVEL_DEBUG = 10000;
static constexpr int LEVEL_TRACE = 100000;

int& get_log_level();

void set_log_level(int lev);

bool will_log_for(int lev);

inline std::string format_log(const char* fmt, va_list& vl)
{
  va_list vl_copy;
  va_copy(vl_copy, vl);
  int length = std::vsnprintf(nullptr, 0, fmt, vl_copy);
  assert(length >= 0);
  std::vector<char> buf(length + 1);
  (void)std::vsnprintf(buf.data(), length + 1, fmt, vl);
  return std::string(buf.data());
}

inline std::string format_log(const char* fmt, ...)
{
  va_list vl;
  va_start(vl, fmt);
  std::string str = format_log(fmt, vl);
  va_end(vl);
  return str;
}

#define SET_ERROR_MSG(msg, location_prefix, fmt, ...)                                            \
  do {                                                                                           \
    int const size1 = std::snprintf(nullptr, 0, "%s", location_prefix);                          \
    int const size2 = std::snprintf(nullptr, 0, "file=%s line=%d: ", __FILE__, __LINE__);        \
    int const size3 = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__);                             \
    if (size1 < 0 || size2 < 0 || size3 < 0) {                                                   \
      (void)printf("Error in snprintf, cannot handle exception.\n");                             \
      (void)fflush(stdout);                                                                      \
      abort();                                                                                   \
    }                                                                                            \
    auto tot_size = size1 + size2 + size3 + 1; /* +1 for final '\0' */                           \
    std::vector<char> buf(tot_size);                                                             \
    (void)std::snprintf(buf.data(), size1 + 1 /* +1 for '\0' */, "%s", location_prefix);         \
    (void)std::snprintf(                                                                         \
      buf.data() + size1, size2 + 1 /* +1 for '\0' */, "file=%s line=%d: ", __FILE__, __LINE__); \
    (void)std::snprintf(                                                                         \
      buf.data() + size1 + size2, size3 + 1 /* +1 for '\0' */, fmt, ##__VA_ARGS__);              \
    msg += std::string(buf.data(), buf.data() + tot_size - 1); /* -1 to remove final '\0' */     \
  } while (0)

// std::cout << format_log(fmt, ##__VA_ARGS__) << std::endl << std::flush;

#define COMMON_LOG(lev, fmt, ...)                                                      \
  do {                                                                                 \
    if (will_log_for(lev))                                                             \
      printf("%s\n", format_log(fmt, ##__VA_ARGS__).c_str());                          \
      fflush(stdout);                                                                  \
  } while (0)

#define LOG_FATAL(fmt, ...)                                                            \
  do {                                                                                 \
    std::string fatal_msg{};                                                           \
    SET_ERROR_MSG(fatal_msg, "FATAL at ", fmt, ##__VA_ARGS__);                         \
    COMMON_LOG(LEVEL_FATAL, "%s", fatal_msg.c_str());                                  \
    abort();                                                                           \
  } while (0)

#define LOG_ERROR(fmt, ...) COMMON_LOG(LEVEL_ERROR, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  COMMON_LOG(LEVEL_WARN, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  COMMON_LOG(LEVEL_INFO, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) COMMON_LOG(LEVEL_DEBUG, fmt, ##__VA_ARGS__)
#define LOG_TRACE(fmt, ...) COMMON_LOG(LEVEL_TRACE, fmt, ##__VA_ARGS__)

