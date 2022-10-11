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
#include <chrono>
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
  bool operator()(const Edge &a, const Edge &b) {
    return a.from < b.from || a.from == b.from && a.to < b.to;
  }
};
void put_edges_to_file(std::string name, std::vector<Edge> &e) {
  // printf("%s\n", name.c_str());
  // for (int i = 0; i < e.size(); i++) {
  //   printf("%d %d\n", e[i].from, e[i].to);
  // }
  std::ofstream ofile;
  std::string file_name = "./" + name + ".txt";
  ofile.open(file_name.c_str());
  for (auto edge : e) {
    ofile << edge.from << "\t" << edge.to << std::endl;
  }
  ofile.close();
}
void generate_random_edge_input(std::vector<std::string> &edge_type,
                                std::vector<std::string> &node_type,
                                int node_each,
                                int edge_num,
                                std::map<Edge, int, EdgeLess> &edge_map,
                                std::vector<std::vector<Edge>> &edge_list) {
  srand(time(0));
  int s = 0;
  std::vector<int> from_type, to_type;
  std::map<std::string, int> node_type_map;
  for (int i = 0; i < node_type.size(); i++) {
    node_type_map[node_type[i]] = i;
  }
  for (auto str : edge_type) {
    auto pos = str.find("2");
    auto from_part = str.substr(0, pos);
    auto to_part = str.substr(pos + 1, str.size() - pos - 1);
    from_type.push_back(node_type_map[from_part]);
    to_type.push_back(node_type_map[to_part]);
  }
  int node_type_size = node_type_map.size();
  int edge_type_size = edge_type.size();
  int total_num = node_each * node_type_size;
  if (node_type_size == 0) return;
  edge_list.resize(edge_type_size);
  int empty_set_num = edge_type_size;
  int type_index;
  while (edge_num > 0 || empty_set_num != 0) {
    Edge temp;

    type_index = rand() % edge_type_size;
    temp.from = from_type[type_index] * node_each + rand() % node_each;
    temp.to = to_type[type_index] * node_each + rand() % node_each;

    if (edge_map.find(temp) != edge_map.end()) continue;
    edge_map[temp] = type_index;
    edge_list[type_index].push_back(temp);
    if (edge_list[type_index].size() == 1) empty_set_num--;
    edge_num--;
  }
  for (int i = 0; i < edge_type_size; i++) {
    put_edges_to_file(edge_type[i], edge_list[i]);
  }
}

TEST(TEST_FLEET, test_heter_graph) {
  auto iter = paddle::framework::GraphGpuWrapper::GetInstance();
  std::vector<int> device;
  int device_num = 2;
  for (int i = 0; i < device_num; i++) device.push_back(i);
  iter->set_device(device);
  const int edge_type_num = 9;
  const int node_type_num = 4;
  std::string edge_type_strs[edge_type_num] = {
      "a2b", "a2c", "b2c", "b2b", "c2a", "c2c", "d2a", "d2c", "d2d"};
  std::string node_type_strs[node_type_num] = {"a", "b", "c", "d"};
  std::vector<std::string> edge_types(edge_type_strs,
                                      edge_type_strs + edge_type_num);
  std::vector<std::string> node_types(node_type_strs,
                                      node_type_strs + node_type_num);
  int node_each = 50000;
  int sample_size = 5;
  int edge_num = (int64_t)node_each * node_each * edge_types.size() * 0.001;
  std::map<Edge, int, EdgeLess> edge_map;
  std::vector<std::vector<Edge>> edge_list;
  generate_random_edge_input(
      edge_types, node_types, node_each, edge_num, edge_map, edge_list);
  std::cerr << "begin to set up types" << std::endl;
  iter->set_up_types(edge_types, node_types);
  std::cerr << "init_service" << std::endl;
  iter->init_service();
  std::cerr << "init_service_over" << std::endl;
  for (int i = 0; i < edge_types.size(); i++)
    iter->load_edge_file(edge_types[i], "./" + edge_types[i] + ".txt", false);
  for (int i = 0; i < edge_types.size(); i++) {
    VLOG(0) << "upload edge_type " << edge_types[i];
    iter->upload_batch(0, i, device_num, edge_types[i]);
    VLOG(0) << "upload edge_type done " << edge_types[i];
  }
  auto edge_type_graph_ = iter->get_edge_type_graph(0, edge_types.size());
  srand(time(0));

  std::vector<uint64_t> key_vec;
  std::vector<int> node_type_vec;
  int total_range = node_types.size() * node_each;
  for (int i = 0; i < node_types.size() * node_each; i++) {
    key_vec.push_back(i);
    node_type_vec.push_back(i / node_each);
  }

  // if (key_vec.size() == 0) {
  //   key_vec.push_back(0);
  //   node_type_vec.push_back(0);
  // }
  for (int i = 0; i < total_range; i++) {
    int r = rand() % (total_range - i);
    std::swap(key_vec[i], key_vec[i + r]);
    std::swap(node_type_vec[i], node_type_vec[i + r]);
  }
  std::vector<int> edges_split_num;
  int edges_len;
  uint64_t *key;
  cudaMalloc((void **)&key, sizeof(uint64_t) * key_vec.size());
  cudaMemcpy(key,
             key_vec.data(),
             sizeof(uint64_t) * key_vec.size(),
             cudaMemcpyHostToDevice);
  int *d_node_types;
  cudaMalloc((void **)&d_node_types, sizeof(int) * node_type_vec.size());
  cudaMemcpy(d_node_types,
             node_type_vec.data(),
             sizeof(int) * node_type_vec.size(),
             cudaMemcpyHostToDevice);

  VLOG(0) << "begin to sample";
  auto res = iter->sample_neighbor_with_node_type(0,
                                                  key,
                                                  sample_size,
                                                  key_vec.size(),
                                                  edge_type_graph_,
                                                  d_node_types,
                                                  node_types.size(),
                                                  edges_len,
                                                  edges_split_num);
  // ASSERT_EQ(res.size(),0);
  int64_t *d_neighbors_ptr = reinterpret_cast<int64_t *>(res[0]->ptr());
  int64_t *d_index_ptr = reinterpret_cast<int64_t *>(res[1]->ptr());
  int *d_type_ptr = reinterpret_cast<int *>(res[2]->ptr());
  std::vector<uint64_t> h_neighbors, h_index;
  std::vector<int> h_node_type;
  h_neighbors.resize(edges_len);
  h_index.resize(edges_len);
  h_node_type.resize(edges_len);
  cudaMemcpy(h_neighbors.data(),
             d_neighbors_ptr,
             edges_len * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_index.data(),
             d_index_ptr,
             edges_len * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_node_type.data(),
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
    last_edge_type = edge_type;
    ASSERT_LT(i, edges_split_num[edge_type]);
    if (edge_type != 0) {
      ASSERT_GE(i, edges_split_num[edge_type - 1]);
    }
    ASSERT_EQ(tmp_type, h_node_type[i]);
    ASSERT_EQ(query_edge_set.find(e), query_edge_set.end());
    query_edge_set.insert(e);
  }

  // hot_start
  for (int i = 0; i < 100; i++) {
    edges_split_num.clear();
    int64_t neighbor_len = 0;
    iter->SampleNeighbors(0,
                          (int64_t *)key,
                          key_vec.size(),
                          sample_size,
                          edges_split_num,
                          &neighbor_len,
                          edge_types.size(),
                          edge_type_graph_);
  }
  for (int i = 0; i < 100; i++) {
    edges_split_num.clear();
    edges_len = 0;
    iter->sample_neighbor_with_node_type(0,
                                         key,
                                         sample_size,
                                         key_vec.size(),
                                         edge_type_graph_,
                                         d_node_types,
                                         node_types.size(),
                                         edges_len,
                                         edges_split_num);
  }

  auto start2 = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; i++) {
    edges_split_num.clear();
    int64_t neighbor_len = 0;
    iter->SampleNeighbors(0,
                          (int64_t *)key,
                          key_vec.size(),
                          sample_size,
                          edges_split_num,
                          &neighbor_len,
                          edge_types.size(),
                          edge_type_graph_);
  }
  auto end2 = std::chrono::steady_clock::now();
  auto tt =
      std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
  std::cerr << "total time cost with all type query is " << tt.count() << " us"
            << std::endl;

  auto start1 = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; i++) {
    edges_split_num.clear();
    edges_len = 0;
    iter->sample_neighbor_with_node_type(0,
                                         key,
                                         sample_size,
                                         key_vec.size(),
                                         edge_type_graph_,
                                         d_node_types,
                                         node_types.size(),
                                         edges_len,
                                         edges_split_num);
  }

  auto end1 = std::chrono::steady_clock::now();
  auto tt1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
  std::cerr << "total time cost with type info query is " << tt1.count()
            << " us" << std::endl;
}
