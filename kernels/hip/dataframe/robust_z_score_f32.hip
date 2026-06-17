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
extern "C" __global__ void navatala_dataframe_robust_z_score_f32(const float* data, const unsigned int* n, const float* median, const float* mad, float* zScores) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int len = n[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    float med = median[0];
    float m = mad[0];
    float xi = data[gid];
    float diff = (xi - med);
    float denom = (__uint_as_float(0x3fbdc5d6u) * m);
    float z = (diff / denom);
    zScores[gid] = z;
  }
}
