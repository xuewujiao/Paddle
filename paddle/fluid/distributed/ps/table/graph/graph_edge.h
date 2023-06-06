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
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
namespace paddle {
namespace distributed {

class GraphEdgeBlob {
 public:
  GraphEdgeBlob() {}
  virtual ~GraphEdgeBlob() {}
  size_t size() { return id_arr.size(); }
  virtual void add_edge(int64_t id, float weight);
  int64_t get_id(int idx) { return id_arr[idx]; }
  virtual half get_weight(int idx) { return (half)(1.0); }
  std::vector<int64_t>& export_id_array() { return id_arr; }

  // ==== adapt for edge feature === 
  virtual int get_feature_ids(int slot_idx,
                              std::vector<uint64_t> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id) const {  // NOLINT
    return 0;
  }
  virtual int get_float_feature(int slot_idx,
                              std::vector<float> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id) const {  // NOLINT
    return 0;
  }
  // virtual void set_feature(int idx, const std::string &str) {}
  // virtual void set_feature_size(int size) {}
  virtual void shrink_to_fit() {}
  virtual std::string *mutable_feature(int idx);
  virtual std::string *mutable_float_feature(int idx);
  // virtual int get_feature_size() { return 0; }
  // ==== adapt for edge feature === 

 protected:
  std::vector<int64_t> id_arr;
};

class WeightedGraphEdgeBlob : public GraphEdgeBlob {
 public:
  WeightedGraphEdgeBlob() {}
  virtual ~WeightedGraphEdgeBlob() {}
  virtual void add_edge(int64_t id, float weight);
  virtual half get_weight(int idx) { return weight_arr[idx]; }

 protected:
  std::vector<half> weight_arr;
};


class WeightedGraphEdgeBlobWithFeature : public GraphEdgeBlob {
 public:
  WeightedGraphEdgeBlobWithFeature() {}
  virtual ~WeightedGraphEdgeBlobWithFeature() {}
  virtual void add_edge(int64_t id, float weight);
  virtual half get_weight(int idx) { return weight_arr[idx]; }

  // === to adapt ====
  virtual int get_feature_ids(int edge_idx,
                              int slot_idx,
                              std::vector<uint64_t> &feature_id,      // NOLINT
                              std::vector<uint8_t> &slot_id) const {  // NOLINT
    errno = 0;
    size_t num = 0;
    if (edge_idx < this->feature.size()) {
      int offset = this->offset[edge_idx];
      if (slot_idx < static_cast<int>(offset)) {
        const std::string &s = this->feature[edge_idx][slot_idx];
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
  }
  
  virtual int get_float_feature(int edge_idx,
                                int slot_idx,
                                std::vector<float> &float_feature,      // NOLINT
                                std::vector<uint8_t> &slot_id) const {  // NOLINT
    errno = 0;
    size_t num = 0;
    if (edge_idx < this->feature.size()) {
      int offset = this->offset[edge_idx];
      if (offset + slot_idx < static_cast<int>(this->feature[edge_idx].size())) {
        const std::string &s = this->feature[edge_idx][offset + slot_idx];
        const float *feas = (const float *)(s.c_str());
        num = s.length() / sizeof(float);
        CHECK((s.length() % sizeof(float)) == 0)
            << "bad feature_item: [" << s << "]";
        for (size_t i = 0; i < num; ++i) {
          float_feature.push_back(feas[i]);
          slot_id.push_back(slot_idx);
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
    if (idx >= static_cast<int>(this->feature.back().size())) {
      this->feature.back().resize(idx + 1);
    }
    if (idx + 1 > this->offset.back()) this->offset.back() = idx + 1;
    return &(this->feature.back()[idx]);
  }

  virtual std::string *mutable_float_feature(int idx) {
    if (offset + idx >= static_cast<int>(this->feature.back().size())) {
      this->feature.back().resize(offset + idx + 1);
    }
    return &(this->feature.back()[offset + idx]);
  }

  virtual void shrink_to_fit() {
    feature.shrink_to_fit();
    for (auto &edge : feature) {
      edge.shrink_to_fit();
      for (auto& slot: edge) {
        slot.shrink_to_fit();
      }
    }
  }
 // === to adapt ==== 
 protected:
  std::vector<half> weight_arr;
  std::vector<int> offset;
  std::vector<std::vector<std::string>> feature;
};

class GraphEdgeBlobWithFeature : public GraphEdgeBlob {
 public:
  GraphEdgeBlobWithFeature() {}
  virtual ~GraphEdgeBlobWithFeature() {}
  virtual void add_edge(int64_t id, float weight);

 // === to adapt ====
 template <typename T>
 virtual int get_feature_ids(int edge_idx,
                             int slot_idx,
                             std::vector<uint64_t> &feature_id,      // NOLINT
                             std::vector<uint8_t> &slot_id) const {  // NOLINT
    errno = 0;
    size_t num = 0;
    if (edge_idx < this->feature.size()) {
      int offset = this->offset[edge_idx];
      if (slot_idx < static_cast<int>(offset)) {
        const std::string &s = this->feature[edge_idx][slot_idx];
        const uint64_t *feas = (const uint64_t *)(s.c_str());
        num = s.length() / sizeof(uint64_t);
        CHECK((s.length() % sizeof(uint64_t)) == 0)
            << "bad feature_item: [" << s << "]";
        for (size_t i = 0; i < num; ++i) {
          feature_id.push_back(feas[i]);
          slot_id.push_back(slot_idx);
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

  virtual int get_float_feature(int edge_idx,
                                int slot_idx,
                                std::vector<float> &float_feature,      // NOLINT
                                std::vector<uint8_t> &slot_id) const {  // NOLINT
    errno = 0;
    size_t num = 0;

    if (edge_idx < static_cast<int>(this->feature.size())) {
      int offset = this->offset[edge_idx];
      if (offset + slot_idx < static_cast<int>(this->feature[edge_idx].size())) {
        const std::string &s = this->feature[edge_idx][offset + slot_idx];
        const float *feas = (const float *)(s.c_str());
        num = s.length() / sizeof(float);
        CHECK((s.length() % sizeof(float)) == 0)
            << "bad feature_item: [" << s << "]";
        for (size_t i = 0; i < num; ++i) {
          float_feature.push_back(feas[i]);
          slot_id.push_back(slot_idx);
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
    if (idx >= static_cast<int>(this->feature.back().size())) {
      this->feature.back().resize(idx + 1);
    }
    if (idx  + 1 > this->offset.back()) this->offset.back() = idx + 1;
    return &(this->feature.back().[idx]);
  }

  virtual std::string *mutable_float_feature(int idx) {
    if (offset + idx >= static_cast<int>(this->feature.back().size())) {
      this->feature.back().resize(offset + idx + 1);
    }
    return &(this->feature.back().[offset + idx]);
  }
  virtual void shrink_to_fit() {
    feature.shrink_to_fit();
    for (auto &edge : feature) {
      edge.shrink_to_fit();
      for (auto& slot: edge) {
        slot.shrink_to_fit();
      }
    }
  }
 // === to adapt ==== 
 protected:
 std::vector<int> offset;
  std::vector<std::vector<std::string>> feature;
};



}  // namespace distributed
}  // namespace paddle
