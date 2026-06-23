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
extern "C" __global__ void navatala_dataframe_mad_f32(const float* sortedData, const unsigned int* n, const float* median, float* mad) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    unsigned int len = n[0];
    float med = median[0];
    unsigned int q1Idx = (len / 4u);
    unsigned int q3Idx = (len - q1Idx);
    float q1Val = sortedData[q1Idx];
    float q3Val = sortedData[q3Idx];
    float dev1 = abs((q1Val - med));
    float dev3 = abs((q3Val - med));
    float madVal = ((dev1 + dev3) / __uint_as_float(0x40000000u));
    mad[0] = madVal;
  }
}
