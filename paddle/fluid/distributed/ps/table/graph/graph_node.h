// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstring>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/distributed/ps/table/graph/graph_weighted_sampler.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

enum FEATYPE{
  INT64 = 0,
  INT32,
  DOUBLE,
  FLOAT
};


struct Feature {
  char* feature;
  uint32_t dtype;
  uint32_t shape;
  Feature(){}
  Feature(char* fea, uint32_t dtype, uint32_t shape):feature(fea), dtype(dtype), shape(shape) {}

  __host__ __device__ Feature& operator=(const Feature& fea) {
    this->feature = fea.feature; 
    this->shape = fea.shape;
    this->dtype = fea.dtype;
    return *this; 
  }
};

class Node {
 public:
  Node() {}
  explicit Node(uint64_t id) : id(id) {}
  virtual ~Node() {}
  static int id_size, int_size, weight_size;
  uint64_t get_id() { return id; }
  int64_t get_py_id() { return (int64_t)id; }
  void set_id(uint64_t id) { this->id = id; }

  virtual void build_edges(bool is_weighted) {}
  virtual void build_sampler(std::string sample_type) {}
  virtual void add_edge(uint64_t id, float weight) {}
  virtual std::vector<int> sample_k(
      int k, const std::shared_ptr<std::mt19937_64> rng) {
    return std::vector<int>();
  }
  virtual uint64_t get_neighbor_id(int idx) { return 0; }
  virtual float get_neighbor_weight(int idx) { return 1.; }

  virtual int get_size(bool need_feature);
  virtual void to_buffer(char *buffer, bool need_feature);
  virtual void recover_from_buffer(char *buffer);
  virtual std::string get_feature(int idx) { return std::string(""); }
  virtual int get_feature_ids(std::vector<uint64_t> *res) const { return 0; }
  virtual int get_feature_ids(int slot_idx, std::vector<uint64_t> *res) const {
    return 0;
  }
  virtual int get_feature_ids(int slot_idx,
                              std::vector<uint64_t> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id) const {  // NOLINT
    return 0;
  }
  virtual int get_feature_ids(int slot_idx,
                              std::vector<Feature> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id,
                              std::vector<uint64_t> & bytes_size) const {  // NOLINT
    return 0;
  }
  virtual int get_slot_feature_ids(int slot_idx,
                                   std::vector<Feature> &feature_id,      // NOLINT
                                   std::vector<uint8_t> &slot_id,
                                   std::vector<uint64_t> & bytes_size) const {  // NOLINT
    return 0;
  }
  virtual void set_feature(int idx, const std::string &str) {}
  virtual void set_feature_shape(int idx, const int32_t & shape) {}
  virtual void set_feature_dtype(int idx, const std::string &dtype) {}
  virtual void set_feature_size(int size) {}
  virtual void shrink_to_fit() {}
  virtual int get_feature_size() { return 0; }
  virtual size_t get_neighbor_size() { return 0; }
  virtual bool get_is_weighted() { return is_weighted; }

 protected:
  uint64_t id;
  bool is_weighted;
};

class GraphNode : public Node {
 public:
  GraphNode() : Node(), sampler(nullptr), edges(nullptr) {}
  explicit GraphNode(uint64_t id)
      : Node(id), sampler(nullptr), edges(nullptr) {}
  virtual ~GraphNode();
  virtual void build_edges(bool is_weighted);
  virtual void build_sampler(std::string sample_type);
  virtual void add_edge(uint64_t id, float weight) {
    edges->add_edge(id, weight);
  }
  virtual std::vector<int> sample_k(
      int k, const std::shared_ptr<std::mt19937_64> rng) {
    return sampler->sample_k(k, rng);
  }
  virtual uint64_t get_neighbor_id(int idx) { return edges->get_id(idx); }
  virtual float get_neighbor_weight(int idx) { return edges->get_weight(idx); }
  virtual size_t get_neighbor_size() { return edges->size(); }

 protected:
  Sampler *sampler;
  GraphEdgeBlob *edges;
};

class FeatureNode : public Node {
 public:
  FeatureNode() : Node() {}
  explicit FeatureNode(uint64_t id) : Node(id) {}
  virtual ~FeatureNode() {}
  virtual int get_size(bool need_feature);
  virtual void to_buffer(char *buffer, bool need_feature);
  virtual void recover_from_buffer(char *buffer);
  virtual std::string get_feature(int idx) {
    if (idx < static_cast<int>(this->feature.size())) {
      return this->feature[idx];
    } else {
      return std::string("");
    }
  }

  virtual int get_feature_ids(std::vector<uint64_t> *res) const {
    PADDLE_ENFORCE_NOT_NULL(res,
                            paddle::platform::errors::InvalidArgument(
                                "get_feature_ids res should not be null"));
    errno = 0;
    for (auto &feature_item : feature) {
      const uint64_t *feas = (const uint64_t *)(feature_item.c_str());
      size_t num = feature_item.length() / sizeof(uint64_t);
      CHECK((feature_item.length() % sizeof(uint64_t)) == 0)
          << "bad feature_item: [" << feature_item << "]";
      size_t n = res->size();
      res->resize(n + num);
      for (size_t i = 0; i < num; ++i) {
        (*res)[n + i] = feas[i];
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
            "get_feature_ids get errno should be 0, but got %d.", errno));
    return 0;
  }

  virtual int get_feature_ids(int slot_idx, std::vector<uint64_t> *res) const {
    PADDLE_ENFORCE_NOT_NULL(res,
                            paddle::platform::errors::InvalidArgument(
                                "get_feature_ids res should not be null"));
    res->clear();
    errno = 0;
    if (slot_idx < static_cast<int>(this->feature.size())) {
      const std::string &s = this->feature[slot_idx];
      const uint64_t *feas = (const uint64_t *)(s.c_str());

      size_t num = s.length() / sizeof(uint64_t);
      CHECK((s.length() % sizeof(uint64_t)) == 0)
          << "bad feature_item: [" << s << "]";
      res->resize(num);
      for (size_t i = 0; i < num; ++i) {
        (*res)[i] = feas[i];
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
            "get_feature_ids get errno should be 0, but got %d.", errno));
    return 0;
  }

  virtual int get_feature_ids(int slot_idx,
                              std::vector<uint64_t> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id) const {  // NOLINT
    errno = 0;
    size_t num = 0;
    if (slot_idx < static_cast<int>(this->feature.size())) {
      const std::string &s = this->feature[slot_idx];
      const uint64_t *feas = (const uint64_t *)(s.c_str());
      num = s.length() / sizeof(uint64_t);
      CHECK((s.length() % sizeof(uint64_t)) == 0)
          << "bad feature_item: [" << s << "]";
      for (size_t i = 0; i < num; ++i) {
        feature_id.push_back(feas[i]);
        slot_id.push_back(slot_idx);
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
            "get_feature_ids get errno should be 0, but got %d.", errno));
    return num;
  }

  virtual int get_feature_ids(int slot_idx,
                              std::vector<Feature> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id,
                              std::vector<uint64_t> &bytes_size) const {  // NOLINT
    errno = 0;
    size_t num = 0;
    if (slot_idx < static_cast<int>(this->feature.size())) {
      const std::string &s = this->feature[slot_idx]; // 对于没有数据的,s.size() == 0
      auto& dtype = this->feature_dtype[slot_idx];
      auto& shape = this->feature_shape[slot_idx]; // 对于uint64,是1?
      
      if (dtype == "feasign" || dtype == "int64") {
        const uint64_t *feas = (const uint64_t *)(s.c_str());
        num = s.length() / sizeof(uint64_t);
        CHECK((s.length() % sizeof(uint64_t)) == 0)
            << "bad feature_item: [" << s << "]";
        for (size_t i = 0; i < num; ++i) {
          // feature_id.push_back(feas[i]);
          // construct Feature
          // Feature fea;
          // fea.dtype = FEATYPE::INT64;
          // fea.shape = shape;
          // fea.feature = (char*)feas[i];
          slot_id.push_back(slot_idx);
          // feature_id.push_back(fea);
          bytes_size.push_back(shape * sizeof(uint64_t));
          feature_id.emplace_back((char*)feas[i], FEATYPE::INT64, shape);
        }
      } else if (dtype == "int32"){
        const int32_t *feas = (const int32_t *)(s.c_str());
        num = s.length() / sizeof(int32_t);
        CHECK((s.length() % sizeof(int32_t)) == 0)
            << "bad feature_item: [" << s << "]";
        for (size_t i = 0; i < num; ++i) {
          // feature_id.push_back(feas[i]);
          // construct Feature
          // Feature fea;
          // fea.dtype = FEATYPE::INT32;
          // fea.shape = shape;
          // fea.feature = (char*)feas[i];
          slot_id.push_back(slot_idx);
          bytes_size.push_back(shape * sizeof(int32_t));
          // feature_id.push_back(fea);
          feature_id.emplace_back((char*)(feas + i), FEATYPE::INT32, shape);
        }

      } else if (dtype == "float64") {
        const double *feas = (const double *)(s.c_str());
        size_t dense_vals = s.length() / sizeof(double);
        CHECK((s.length() % sizeof(double)) == 0)
            << "bad feature_item: [" << s << "]";
        CHECK(dense_vals == (size_t)shape)
            << "bad feature_item: [" << s << "]";
        // for (size_t i = 0; i < dense_vals; ++i) {
        //  // feature_id.push_back(feas[i]);
        //  // construct Feature
        //  Feature fea;
        //  fea.dtype = dtype;
        //  fea.shape = shape;
        //  fea.feature = (char*)feas[i];
        //  slot_id.push_back(slot_idx);
        //  feature_id.push_back(fea);
        // }
        // Feature fea;
        // fea.dtype = FEATYPE::DOUBLE;
        // fea.shape = shape;
        // fea.feture = (char*)feas
        slot_id.push_back(slot_idx);
        bytes_size.push_back(shape * sizeof(double));
        feature_id.emplace_back((char*)feas, FEATYPE::DOUBLE, shape);
        num = 1;
      } else if (dtype == "float32") {
        const float *feas = (const float *)(s.c_str());
        auto dense_vals = s.length() / sizeof(float);
        CHECK((s.length() % sizeof(float)) == 0)
            << "bad feature_item: [" << s << "]";
        CHECK(dense_vals == (size_t)shape)
            << "bad feature_item: [" << s << "]";
        // for (size_t i = 0; i < dense_vals; ++i) {
        //  // feature_id.push_back(feas[i]);
        //  // construct Feature
        //  Feature fea;
        //  fea.dtype = dtype;
        //  fea.shape = shape;
        //  fea.feature = (char*)feas[i];
        //  slot_id.push_back(slot_idx);
        //  feature_id.push_back(fea);
        // }
        // Feature fea;
        // fea.dtype = FEATYPE::FLOAT;
        // fea.shape = shape;
        // fea.feture = (char*)feas
        slot_id.push_back(slot_idx);
        bytes_size.push_back(shape * sizeof(float));
        feature_id.emplace_back((char*)feas, FEATYPE::FLOAT, shape);
        num = 1;
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
            "get_feature_ids get errno should be 0, but got %d.", errno));
    return num;
  }



  virtual int get_slot_feature_ids(int slot_idx,
                              std::vector<Feature> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id,
                              std::vector<uint64_t> &bytes_size) const {  // NOLINT
    errno = 0;
    size_t num = 0;
    if (slot_idx < static_cast<int>(this->feature.size())) {
      const std::string &s = this->feature[slot_idx]; // 对于没有数据的,s.size() == 0
      auto& dtype = this->feature_dtype[slot_idx];
      auto& shape = this->feature_shape[slot_idx]; // 对于uint64,是1?
     
    
      VLOG(0) << "slot idx:" << slot_idx << ", dtype:" << dtype << ", shape:" << shape;  

 
      if (dtype == "feasign" || dtype == "int64") {
        const uint64_t *feas = (const uint64_t *)(s.c_str());
        num = s.length() / sizeof(uint64_t);
        CHECK((s.length() % sizeof(uint64_t)) == 0)
            << "bad feature_item: [" << s << "]";
        for (size_t i = 0; i < num; ++i) {
          // feature_id.push_back(feas[i]);
          // construct Feature
          // Feature fea;
          // fea.dtype = FEATYPE::INT64;
          // fea.shape = shape;
          // fea.feature = (char*)feas[i];
          slot_id.push_back(slot_idx);
          // feature_id.push_back(fea);
          bytes_size.push_back(shape * sizeof(uint64_t));
          feature_id.emplace_back((char*)feas[i], FEATYPE::INT64, shape);
        }
      } else if (dtype == "int32"){
        const int32_t *feas = (const int32_t *)(s.c_str());
        num = s.length() / sizeof(int32_t);
        CHECK((s.length() % sizeof(int32_t)) == 0)
            << "bad feature_item: [" << s << "]";
        for (size_t i = 0; i < num; ++i) {
          // feature_id.push_back(feas[i]);
          // construct Feature
          // Feature fea;
          // fea.dtype = FEATYPE::INT32;
          // fea.shape = shape;
          // fea.feature = (char*)feas[i];
          slot_id.push_back(slot_idx);
          bytes_size.push_back(shape * sizeof(int32_t));
          // feature_id.push_back(fea);
          feature_id.emplace_back((char*)(feas + i), FEATYPE::INT32, shape);
        }
      }
    }
    PADDLE_ENFORCE_EQ(
        errno,
        0,
        paddle::platform::errors::InvalidArgument(
            "get_feature_ids get errno should be 0, but got %d.", errno));
    return num;
  }








  virtual std::string *mutable_feature(int idx) {
    if (idx >= static_cast<int>(this->feature.size())) {
      this->feature.resize(idx + 1);
    }
    return &(this->feature[idx]);
  }

  virtual void set_feature(int idx, const std::string &str) {
    if (idx >= static_cast<int>(this->feature.size())) {
      this->feature.resize(idx + 1);
    }
    this->feature[idx] = str;
  }
  virtual void set_feature_size(int size) { 
      this->feature.resize(size);
      this->feature_dtype.resize(size);
      this->feature_shape.resize(size);
  }
  virtual void set_feature_shape(int idx, const int32_t & shape) {
    if (idx >= static_cast<int>(this->feature_shape.size())) {
      this->feature_shape.resize(idx + 1);
    }
    this->feature_shape[idx] = shape;
  }
  virtual void set_feature_dtype(int idx, const std::string &dtype) {
    if (idx >= static_cast<int>(this->feature_dtype.size())) {
      this->feature_dtype.resize(idx + 1);
    }
    this->feature_dtype[idx] = dtype;
  }
  virtual int get_feature_size() { return this->feature.size(); }
  virtual void shrink_to_fit() {
    feature.shrink_to_fit();
    for (auto &slot : feature) {
      slot.shrink_to_fit();
    }
  }

  template <typename T>
  static std::string parse_value_to_bytes(std::vector<std::string> feat_str) {
    T v;
    size_t Tsize = sizeof(T) * feat_str.size();
    char buffer[Tsize];
    for (size_t i = 0; i < feat_str.size(); i++) {
      std::stringstream ss(feat_str[i]);
      ss >> v;
      std::memcpy(
          buffer + sizeof(T) * i, reinterpret_cast<char *>(&v), sizeof(T));
    }
    return std::string(buffer, Tsize);
  }

  template <typename T>
  static void parse_value_to_bytes(
      std::vector<std::string>::iterator feat_str_begin,
      std::vector<std::string>::iterator feat_str_end,
      std::string *output) {
    T v;
    size_t feat_str_size = feat_str_end - feat_str_begin;
    size_t Tsize = sizeof(T) * feat_str_size;
    char buffer[Tsize] = {'\0'};
    for (size_t i = 0; i < feat_str_size; i++) {
      std::stringstream ss(*(feat_str_begin + i));
      ss >> v;
      std::memcpy(
          buffer + sizeof(T) * i, reinterpret_cast<char *>(&v), sizeof(T));
    }
    output->assign(buffer);
  }

  template <typename T>
  static std::vector<T> parse_bytes_to_array(std::string feat_str) {
    T v;
    std::vector<T> out;
    size_t start = 0;
    const char *buffer = feat_str.data();
    while (start < feat_str.size()) {
      std::memcpy(reinterpret_cast<char *>(&v), buffer + start, sizeof(T));
      start += sizeof(T);
      out.push_back(v);
    }
    return out;
  }

  template <typename T>
  static int parse_value_to_bytes(
      std::vector<paddle::string::str_ptr>::iterator feat_str_begin,
      std::vector<paddle::string::str_ptr>::iterator feat_str_end,
      std::string *output) {
    size_t feat_str_size = feat_str_end - feat_str_begin;
    size_t Tsize = sizeof(T) * feat_str_size;
    size_t num = output->length();
    output->resize(num + Tsize);

    T *fea_ptrs = reinterpret_cast<T *>(&(*output)[num]);

    thread_local paddle::string::str_ptr_stream ss;
    for (size_t i = 0; i < feat_str_size; i++) {
      ss.reset(*(feat_str_begin + i));
      int len = ss.end - ss.ptr;
      char *old_ptr = ss.ptr;
      ss >> fea_ptrs[i];
      if (ss.ptr - old_ptr != len) {
        return -1;
      }
    }
    return 0;
  }

 protected:
  std::vector<std::string> feature;
  std::vector<int32_t> feature_shape;
  std::vector<std::string> feature_dtype;
  
};

}  // namespace distributed
}  // namespace paddle
