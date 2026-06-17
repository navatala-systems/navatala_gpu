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
extern "C" __global__ void navatala_vector_search_decode_p_q_vectors_f32(const unsigned char* codes, const float* codebook, const unsigned int* n_vectors, const unsigned int* sub_dim, float* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nv = n_vectors[0];
  unsigned int sd = sub_dim[0];
  unsigned int total = (nv * sd);
  if ((gid < total)) {
    unsigned int vid = (gid / sd);
    unsigned int k = (gid % sd);
    unsigned char code = codes[vid];
    unsigned int cw = ((unsigned int)(code));
    unsigned int cb_idx = ((cw * sd) + k);
    float val = codebook[cb_idx];
    _output[gid] = val;
  }
}
