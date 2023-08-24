#include "log_macros.h"

int& get_log_level()
{
  static int log_level = LEVEL_INFO;
  return log_level;
}

void set_log_level(int lev) { get_log_level() = lev; }

bool will_log_for(int lev) { return lev <= get_log_level(); }
