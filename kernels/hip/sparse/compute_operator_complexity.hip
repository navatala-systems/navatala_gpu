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
extern "C" __global__ void navatala_sparse_compute_operator_complexity(const unsigned int* levelNnz, const unsigned int* levelRows, const unsigned int* nLevels, float* operatorComplexity) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    int nLev = ((int)(nLevels[0]));
    unsigned int finestNnz = levelNnz[0];
    float total = __uint_as_float(0x00000000u);
    for (int lev = 0; lev < (int)(nLev); ++lev) {
      unsigned int nnzL = levelNnz[lev];
      total = (total + ((float)(nnzL)));
    }
    float oc = (total / ((float)(finestNnz)));
    operatorComplexity[0] = oc;
  }
}
