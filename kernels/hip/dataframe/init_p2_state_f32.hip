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
extern "C" __global__ void navatala_dataframe_init_p2_state_f32(const float* p, float* q0, float* q1, float* q2, float* q3, float* q4, unsigned int* pos0, unsigned int* pos1, unsigned int* pos2, unsigned int* pos3, unsigned int* pos4, float* targetP, unsigned int* count) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    float pVal = p[0];
    q0[0] = __uint_as_float(0x00000000u);
    q1[0] = __uint_as_float(0x00000000u);
    q2[0] = __uint_as_float(0x00000000u);
    q3[0] = __uint_as_float(0x00000000u);
    q4[0] = __uint_as_float(0x00000000u);
    pos0[0] = 0u;
    pos1[0] = 0u;
    pos2[0] = 0u;
    pos3[0] = 0u;
    pos4[0] = 0u;
    targetP[0] = pVal;
    count[0] = 0u;
  }
}
