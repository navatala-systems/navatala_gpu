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
extern "C" __global__ void navatala_dataframe_init_m_c_d_f32(const float* thresholdIn, float* robustMeanX, float* robustMeanY, float* robustCov, float* determinant, float* threshold, unsigned int* count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    float thresh = thresholdIn[0];
    robustMeanX[0] = __uint_as_float(0x00000000u);
    robustMeanY[0] = __uint_as_float(0x00000000u);
    robustCov[0] = __uint_as_float(0x00000000u);
    determinant[0] = __uint_as_float(0x00000000u);
    threshold[0] = thresh;
    count[0] = 0u;
  }
}
