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
extern "C" __global__ void navatala_dataframe_extract_skewness_f32(const unsigned int* n, const float* M2, const float* M3, float* skewness) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0)) {
    unsigned int nVal = n[0];
    float m2Val = M2[0];
    float m3Val = M3[0];
    float nf = ((float)(nVal));
    float sqrtN = sqrt(nf);
    float sqrtM2 = sqrt(m2Val);
    float m2Pow15 = (m2Val * sqrtM2);
    float denomSafe = (((m2Val == __uint_as_float(0x00000000u))) ? (__uint_as_float(0x3f800000u)) : (m2Pow15));
    float numerator = (sqrtN * m3Val);
    float skewRaw = (numerator / denomSafe);
    float skewResult = (((m2Val == __uint_as_float(0x00000000u))) ? (__uint_as_float(0x00000000u)) : (skewRaw));
    skewness[0] = skewResult;
  }
}
