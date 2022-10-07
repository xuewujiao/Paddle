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

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_ps_table.h"
#include "paddle/fluid/framework/fleet/heter_ps/graph_gpu_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_comm.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/fleet/heter_ps/optimizer.cuh.h"
#include "paddle/fluid/platform/cuda_device_guard.h"

using namespace paddle::framework;
namespace platform = paddle::platform;
struct Edge {
  uint64_t from, to;
};
struct EdgeLess {
  bool operator<(const Edge &a, const Edge &b) {
    return a.from < b.from || a.from == b.from && a.to < b.to;
  }
} void put_edges_to_file(std::string name, std::vector<Edge> &e) {
  printf("%s\n", name.c_str());
  for (int i = 0; i < e.size(); i++) {
    printf("%d %d\n", e.from, e.to);
  }
  std::ofstream ofile;
  std::file_name = "./" + name + ".txt";
  ofile.open(file_name);
  for (auto edge : e) {
    ofile << e.from << "\t" << e.to << std::endl;
  }
  ofile.close();
}
void generate_random_edge_input(std::vector<std::string> &edge_type,
                                int node_each,
                                int edge_num,
                                std::map<Edge, int, EdgeLess> &edge_map,
                                std::vector<Edge> &edge_list) {
  srand(time(0));
  int s = 0;
  std::vector<Edge> edge_list;
  std::vector<int> from_type, to_type;
  std::map<std::string, int> node_type_set;
  for (auto str : edge_type) {
    auto pos = edge_type.find("2");
    auto from_part = edge_type.substr(0, pos);
    auto to_part = edge_type.substr(pos + 1, edge_type.size() - pos - 1);
    if (node_type_set.find(from_part) == node_type_set.end())
      node_type_set[from_part] = node_type_set.size();
    if (node_type_set.find(to_part) == node_type_set.end())
      node_type_set[to_part] = node_type_set.size();
    from_type.push_back(node_type_set[from_part]);
    to_type.push_back(node_type_set[to_part]);
  }
  s = node_type_set.size();
  int total_num = node_each * s;
  if (s == 0) return;
  std::set<Edge, EdgeLess> edge_set;
  edge_list.resize(s);
  int empty_set_num = s;
  int type_index;
  while (edge_num > 0 || empty_set_num != 0) {
    Edge temp;
    do {
      type_index = rand() % s;
      temp.from = from[type_index] * node_each + rand() % node_each;
      temp.to = to[type_index] * node_each + rand() % node_each;

        }while(edge_set.find(temp) != edge_set.end()));
        edge_set[temp] = type_index;
        edge_list[type_index].push_back(temp);
        if (edge_list[type_index].size() == 1) empyt_set_num--;
        edge_num--;
  }
  for (int i = 0; i < edge_type.size(); i++) {
    put_edges_to_file(edge_type[i], edge_list[i]);
  }
}

char edge_file_name[] = "edges1.txt";

char node_file_name[] = "nodes.txt";
void prepare_file(char *file_name, bool load_edge) {
  std::ofstream ofile;
  ofile.open(file_name);
  if (load_edge) {
    for (auto x : edges) {
      ofile << x << std::endl;
    }
  } else {
    for (auto x : nodes) {
      ofile << x << std::endl;
    }
  }
  ofile.close();
}
TEST(TEST_FLEET, test_heter_graph) {
  auto iter = paddle::framework::GraphGpuWrapper::GetInstance();
  std::vector<int> device;
  device.push_back(0);
  device.push_back(1);
  iter->set_device(device);
  std::vector < std::string >> edge_types = {"a2b", "a2c", "b2c", "b2b"};
  std::vector < std::string >> node_types = {"a", "b", "c"};
  int node_each = 3;
  std::set<Edge, EdgeLess> edge_map;
  std::vector<Edge> edge_list;
  generate_random_edge_input(edge_types, node_each, 20, edge_map, edge_list);
  iter->set_up_types(edge_types, node_types);
  iter->init_service();
  auto edge_type_graph_ =
      gpu_graph_ptr->get_edge_type_graph(gpuid_, edge_to_id_len_);
  srand(time(0));

  std::vector<uint64_t> key_vec;
  std::vector<int> node_type_vec;
  for (int i = 0; i < 9; i++) {
    if (rand() % 2) {
      key_vec.push_back(i);
      node_type_vec.push_back(i / 3);
    }
  }
  if (key_vec.size() == 0) {
    key_vec.push_back(0);
    node_type_vec.push_back(0);
  }
  std::vector<int> edges_split_num;
  int edges_len;
  uint64_t *key;
  cudaMalloc((void **)&key, sizeof(uint64_t) * key_vec.size());
  cudaMemcpy(key,
             key_vec.begin(),
             sizeof(uint64_t) * key_vec.size(),
             cudaMemcpyHostToDevice);
  int *node_types;
  cudaMalloc((void **)&node_types, sizeof(int) * node_type_vec.size());
  cudaMemcpy(node_types,
             node_type_vec.begin(),
             sizeof(int) * node_type_vec.size(),
             cudaMemcpyHostToDevice);
  q1.initialize(i, 0, (uint64_t)key, 2, 1);

  auto res = gpu_graph_ptr->sample_neighbor_with_node_type(0,
                                                           key,
                                                           2,
                                                           key_vec.size(),
                                                           edge_type_graphs,
                                                           node_types,
                                                           3,
                                                           edges_len,
                                                           edges_split_num);
  int64_t *d_neighbors_ptr = reinterpret_cast<int *>(res[0]->ptr());
  int64_t *d_index_ptr = reinterpret_cast<int *>(res[1]->ptr());
  int *d_type_ptr = reinterpret_cast<int *>(res[2]->ptr());
  std::vector<uint64_t> h_neighbors, h_index;
  vector<int> h_node_type;
  h_neighbors.resize(edges_len);
  h_index.resize(edges_len);
  h_node_type.resize(edges_len);
  CudaMemcpy(h_neighbors.data(),
             d_neighbors_ptr,
             edges_len * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  CudaMemcpy(h_index.data(),
             d_index_ptr,
             edges_len * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  CudaMemcpy(h_node_type.data(),
             d_type_ptr,
             edges_len * sizeof(int),
             cudaMemcpyDeviceToHost);
  std::set<Edge, EdgeLess> query_edge_set;
  int last_edge_type = -1;
  for (int i = 0; i < edges_len; i++) {
    ASSERT_LT((size_t)h_index[i], key_vec.size());
    uint64_t from = key_vec[h_index[i]];
    Edge e;
    e.from = from;
    e.to = h_neighbors[i];
    ASSERT_NE(edge_map.find(e), edge_map.end());
    int edge_type = edge_map[e];
    int tmp_type = e.to / node_each;
    ASSERT_GE(edge_type, last_edge_type);
    ASSERT_LT(edge_type, edge_types.size());
    last_edge_type = edge_type;
    ASSERT_LT(i, edges_split_num[edge_type]);
    if (edge_type != 0) {
      ASSERT_GE(i, edges_split_num[edge_type - 1]);
    }
    ASSERT_EQ(tmp_type, h_node_type[i]);
    ASSERT_EQ(query_edge_set.find(e), query_edge_set.end());
    query_edge_set.insert(e);
  }
  return 0;
}
