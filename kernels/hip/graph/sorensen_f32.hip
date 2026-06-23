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
extern "C" __global__ void navatala_graph_sorensen_f32(const unsigned int* inter, const unsigned int* degrees, const unsigned int* pairsA, const unsigned int* pairsB, const unsigned int* numPairs, float* coeff) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int numP = numPairs[0];
  if (gid < numP) {
    unsigned int a = pairsA[gid];
    unsigned int b = pairsB[gid];
    unsigned int iU = inter[gid];
    unsigned int dAU = degrees[a];
    unsigned int dBU = degrees[b];
    float iF = ((float)(iU));
    float dAF = ((float)(dAU));
    float dBF = ((float)(dBU));
    coeff[gid] = ((((dAF + dBF) > __uint_as_float(0x00000000u))) ? (((__uint_as_float(0x40000000u) * iF) / (dAF + dBF))) : (__uint_as_float(0x00000000u)));
  }
}
