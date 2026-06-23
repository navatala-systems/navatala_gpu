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
extern "C" __global__ void navatala_dataframe_iqm_f32(const float* sortedData, const unsigned int* n, float* iqm) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    unsigned int len = n[0];
    unsigned int q1Idx = (len / 4u);
    unsigned int q3Idx = (len - q1Idx);
    unsigned int midIdx = ((q1Idx + q3Idx) / 2u);
    float midVal = sortedData[midIdx];
    iqm[0] = midVal;
  }
}
