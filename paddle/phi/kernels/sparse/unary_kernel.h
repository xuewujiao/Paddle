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

#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/sparse_csr_tensor.h"

namespace phi {
namespace sparse {

#define DECLARE_SPARSE_UNARY_KERNEL(prefix)                                    \
  template <typename T, typename Context>                                      \
  void prefix##CooKernel(                                                      \
      const Context& dev_ctx, const SparseCooTensor& x, SparseCooTensor* out); \
                                                                               \
  template <typename T, typename Context>                                      \
  void prefix##CsrKernel(                                                      \
      const Context& dev_ctx, const SparseCsrTensor& x, SparseCsrTensor* out);

#define DECLARE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(prefix, attr) \
  template <typename T, typename Context>                       \
  void prefix##CooKernel(const Context& dev_ctx,                \
                         const SparseCooTensor& x,              \
                         float attr,                            \
                         SparseCooTensor* out);                 \
                                                                \
  template <typename T, typename Context>                       \
  void prefix##CsrKernel(const Context& dev_ctx,                \
                         const SparseCsrTensor& x,              \
                         float attr,                            \
                         SparseCsrTensor* out);

DECLARE_SPARSE_UNARY_KERNEL(Sin)
DECLARE_SPARSE_UNARY_KERNEL(Tan)
DECLARE_SPARSE_UNARY_KERNEL(Asin)
DECLARE_SPARSE_UNARY_KERNEL(Atan)
DECLARE_SPARSE_UNARY_KERNEL(Sinh)
DECLARE_SPARSE_UNARY_KERNEL(Asinh)
DECLARE_SPARSE_UNARY_KERNEL(Atanh)
DECLARE_SPARSE_UNARY_KERNEL(Relu)
DECLARE_SPARSE_UNARY_KERNEL(Tanh)
DECLARE_SPARSE_UNARY_KERNEL(Square)
DECLARE_SPARSE_UNARY_KERNEL(Sqrt)
DECLARE_SPARSE_UNARY_KERNEL(Log1p)
DECLARE_SPARSE_UNARY_KERNEL(Abs)
DECLARE_SPARSE_UNARY_KERNEL_WITH_ONE_ATTR(Pow, factor)

template <typename T, typename Context>
void ScaleCooKernel(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    float scale,
                    float bias,
                    bool bias_after_scale,
                    SparseCooTensor* out);

template <typename T, typename Context>
void ScaleCsrKernel(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    float scale,
                    float bias,
                    bool bias_after_scale,
                    SparseCsrTensor* out);

template <typename T, typename Context>
void DivCooScalarKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        float scalar,
                        SparseCooTensor* out);

template <typename T, typename Context>
void DivCsrScalarKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        float scalar,
                        SparseCsrTensor* out);

template <typename T, typename Context>
void CastCooKernel(const Context& dev_ctx,
                   const SparseCooTensor& x,
                   DataType index_dtype,
                   DataType value_dtype,
                   SparseCooTensor* out);

template <typename T, typename Context>
void CastCsrKernel(const Context& dev_ctx,
                   const SparseCsrTensor& x,
                   DataType index_dtype,
                   DataType value_dtype,
                   SparseCsrTensor* out);

template <typename T, typename Context>
SparseCooTensor ReluCoo(const Context& dev_ctx, const SparseCooTensor& x) {
  SparseCooTensor coo;
  ReluCooKernel<T, Context>(dev_ctx, x, &coo);
  return coo;
}

template <typename T, typename Context>
SparseCooTensor ReluCsr(const Context& dev_ctx, const SparseCooTensor& x) {
  SparseCooTensor csr;
  ReluCsrKernel<T, Context>(dev_ctx, x, &csr);
  return csr;
}

}  // namespace sparse
}  // namespace phi
