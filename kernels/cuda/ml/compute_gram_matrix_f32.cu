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
extern "C" __global__ void navatala_ml_compute_gram_matrix_f32(const float* X, const unsigned int* nSamples, const unsigned int* nFeatures, float* G) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nSamples[0];
  unsigned int p = nFeatures[0];
  unsigned int totalElems = (p * p);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / p);
    unsigned int j = (gid % p);
    float dotProduct = __uint_as_float(0x00000000u);
    unsigned int idx0_i = i;
    unsigned int idx0_j = j;
    float x0_i = X[idx0_i];
    float x0_j = X[idx0_j];
    float prod0 = (x0_i * x0_j);
    float sum = prod0;
    G[gid] = sum;
  }
}
