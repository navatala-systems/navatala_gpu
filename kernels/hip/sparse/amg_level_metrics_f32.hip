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
extern "C" __global__ void navatala_sparse_amg_level_metrics_f32(const unsigned int* levelNRows, const unsigned int* levelNnz, const unsigned int* nLevels, float* gridComplexity, float* operatorComplexity) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nL = ((int)(nLevels[0]));
  float rows0 = ((float)(levelNRows[0]));
  float nnz0 = ((float)(levelNnz[0]));
  float totalRows = __uint_as_float(0x00000000u);
  float totalNnz = __uint_as_float(0x00000000u);
  for (int i = 0; i < (int)(nL); ++i) {
    float ri = ((float)(levelNRows[i]));
    float ni = ((float)(levelNnz[i]));
    totalRows = (totalRows + ri);
    totalNnz = (totalNnz + ni);
  }
  gridComplexity[0] = (totalRows / rows0);
  operatorComplexity[0] = (totalNnz / nnz0);
}
