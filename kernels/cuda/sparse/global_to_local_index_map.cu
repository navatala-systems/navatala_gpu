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
extern "C" __global__ void navatala_sparse_global_to_local_index_map(const unsigned int* globalIndices, const unsigned int* localOffset, const unsigned int* nLocal, const unsigned int* nIndices, unsigned int* localIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = ((int)(nIndices[0]));
  if ((gid < N)) {
    unsigned int gIdx = globalIndices[gid];
    unsigned int offset = localOffset[0];
    unsigned int nLoc = nLocal[0];
    unsigned int localIdx = (gIdx - offset);
    bool valid = (localIdx < nLoc);
    localIndices[gid] = ((valid) ? (localIdx) : (4294967295u));
  }
}
