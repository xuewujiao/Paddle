#pragma once

#include <string>
#include <vector>

struct Config {
  int node_count = -1;
  std::string sideband_server_name;
  int sideband_server_port=-1;
  std::vector<int> agent_local_rank;
  std::vector<std::string> ib_device_name;
  std::vector<int> ib_port;

  int GetNodeCount();
  int GetRanksPerNode();
};

void ParseConfigFile(const std::string& file_name, Config* config);