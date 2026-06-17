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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_vector_search_scalar_quantize_f32_to_i8(const float* _input, const unsigned int* n_vectors, const unsigned int* dim, const float* mins, const float* maxs, signed char* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int d = dim[0];
  unsigned int total = (nv * d);
  if ((gid < total)) {
    unsigned int dim_idx = (gid % d);
    float val = _input[gid];
    float min_val = mins[dim_idx];
    float max_val = maxs[dim_idx];
    float range = (max_val - min_val);
    float normalized = ((val - min_val) / range);
    float scaled = (normalized * __uint_as_float(0x437f0000u));
    float offset = (scaled - __uint_as_float(0x43000000u));
    float neg128 = (__uint_as_float(0x00000000u) - __uint_as_float(0x43000000u));
    float clamped = (((offset < neg128)) ? (neg128) : ((((offset > __uint_as_float(0x42fe0000u))) ? (__uint_as_float(0x42fe0000u)) : (offset))));
    int quantized_i32 = ((int)(clamped));
    signed char quantized = ((signed char)(quantized_i32));
    _output[gid] = quantized;
  }
}
