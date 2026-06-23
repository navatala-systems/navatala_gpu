// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_vector_search_scalar_dequantize_i8_to_f32(const signed char* _input, const unsigned int* n_vectors, const unsigned int* dim, const float* mins, const float* maxs, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int d = dim[0];
  unsigned int total = (nv * d);
  if (gid < total) {
    unsigned int dim_idx = (gid % d);
    signed char qval = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float qval_f32 = ((float)(qval));
    float offset = (qval_f32 + __uint_as_float(0x43000000u));
    float normalized = (offset / __uint_as_float(0x437f0000u));
    float range = (max_val - min_val);
    float dequant = ((normalized * range) + min_val);
    _output[gid] = dequant;
  }
}
