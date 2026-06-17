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
#include <hip/hip_fp16.h>
extern "C" __global__ void navatala_transformer_generate_causal_mask_f16(const unsigned int* seqLen, __half* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int sl = seqLen[0u];
  unsigned int total = (sl * sl);
  bool inBounds = (gid < total);
  if (inBounds) {
    unsigned int row = (gid / sl);
    unsigned int col = (gid % sl);
    bool canAttend = (col <= row);
    if (canAttend) {
      _output[gid] = __float2half(0.000000f);
    } else {
      _output[gid] = __float2half(-65000.000000f);
    }
  }
}
