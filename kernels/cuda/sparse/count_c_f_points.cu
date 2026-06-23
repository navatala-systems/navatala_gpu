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
extern "C" __global__ void navatala_sparse_count_c_f_points(const int* cfMarking, const unsigned int* nRows, unsigned int* nCoarse, unsigned int* nFine) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nRows[0]));
  if (gid < N) {
    int mark = cfMarking[gid];
    if (mark == 1) {
      unsigned int _nc = atomicAdd(&(nCoarse[0]), 1u);
    }
    if (mark == -1) {
      unsigned int _nf = atomicAdd(&(nFine[0]), 1u);
    }
  }
}
