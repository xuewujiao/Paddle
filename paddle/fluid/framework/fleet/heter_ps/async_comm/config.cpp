#include "config.h"

#include <fstream>
#include <sstream>

#include "check_macros.h"
#include "log_macros.h"

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream token_stream(s);
  while (std::getline(token_stream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

void ParseConfigFile(const std::string &file_name,
                     Config* config) {
  std::ifstream config_file(file_name);

  if (!config_file.is_open()) {
    LOG_FATAL("Cannot open config file");
  }

  std::string line;
  while (std::getline(config_file, line)) {
    std::istringstream iss(line);
    std::string key, value;
    if (std::getline(iss, key, '=') && std::getline(iss, value)) {
      if (key == "ib_device_name") {
        config->ib_device_name = split(value, ',');
      } else if (key == "agent_local_rank") {
        for (const std::string &agent_str: split(value, ',')) {
          config->agent_local_rank.push_back(std::stoi(agent_str));
        }
      } else if (key == "ib_port") {
        for (const std::string &port_str: split(value, ',')) {
          config->ib_port.push_back(std::stoi(port_str));
        }
      } else if (key == "node_count") {
        config->node_count = std::stoi(value);
      } else if (key == "sideband_server_port") {
        config->sideband_server_port = std::stoi(value);
      } else if (key == "sideband_server_name") {
        config->sideband_server_name = value;
      } else {
        LOG_FATAL("Unknown key in config file: %s", key.c_str());
      }
    }
  }
  auto ranks_per_node = config->agent_local_rank.size();
  BOOL_CHECK(config->agent_local_rank.size() == ranks_per_node);
  BOOL_CHECK(config->ib_device_name.size() == ranks_per_node);
  BOOL_CHECK(config->ib_port.size() == ranks_per_node);
}

int Config::GetNodeCount() {
  return node_count;
}
int Config::GetRanksPerNode() {
  return agent_local_rank.size();
}
