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
extern "C" __global__ void navatala_cfd_prolongate_by_agglomeration_f32(const float* coarseField, const unsigned int* fineToCoarse, const unsigned int* nFineCells, float* fineField) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nFineCells[0]));
  if (i < N) {
    int coarseIdx = ((int)(fineToCoarse[i]));
    float val = coarseField[coarseIdx];
    float cur = fineField[i];
    fineField[i] = (cur + val);
  }
}
