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
extern "C" __global__ void navatala_sparse_sparse_threshold_f32(const float* valuesA, const float* threshold, const int* nnz, float* valuesB) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = ((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  float thresh = threshold[0u];
  if (i < nnz[0u]) {
    float a = valuesA[i];
    float absA = abs(a);
    float result = (((absA >= thresh)) ? (a) : (__uint_as_float(0x00000000u)));
    valuesB[i] = result;
  }
}
