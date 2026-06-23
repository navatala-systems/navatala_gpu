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
extern "C" __global__ void navatala_dataframe_init_t_digest_f32(const float* compressionIn, float* centroidMeans, float* centroidWeights, unsigned int* centroidCount, float* totalWeight, float* minVal, float* maxVal, float* compression) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    float delta = compressionIn[0];
    centroidCount[0] = 0u;
    totalWeight[0] = __uint_as_float(0x00000000u);
    minVal[0] = __uint_as_float(0x7f7ffffdu);
    maxVal[0] = __uint_as_float(0xff7ffffdu);
    compression[0] = delta;
  }
}
