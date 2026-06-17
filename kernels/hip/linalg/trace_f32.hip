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
extern "C" __global__ void navatala_linalg_trace_f32(const float* A, const unsigned int* n, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int dim = n[0u];
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    float traceAccum = __uint_as_float(0x00000000u);
    for (int i = 0; i < (int)(dim); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      unsigned int diagIdx = ((iU32 * dim) + iU32);
      float diagVal = A[diagIdx];
      float currentTrace = traceAccum;
      float nextTrace = (currentTrace + diagVal);
      traceAccum = nextTrace;
    }
    float finalTrace = traceAccum;
    result[0u] = finalTrace;
  }
}
