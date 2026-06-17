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
extern "C" __global__ void navatala_graph_normalized_laplacian_f32(const float* W, const float* D, const unsigned int* n, float* L_sym) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nVal = n[0];
  unsigned int totalElems = (nVal * nVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    unsigned int i = (gid / nVal);
    unsigned int j = (gid % nVal);
    float di = D[i];
    float dj = D[j];
    float eps = __uint_as_float(0x2edbe6ffu);
    bool diZero = (di < eps);
    bool djZero = (dj < eps);
    bool isolated = (diZero || djZero);
    bool isDiag = (i == j);
    if (isolated) {
      if (isDiag) {
        L_sym[gid] = __uint_as_float(0x3f800000u);
      } else {
        L_sym[gid] = __uint_as_float(0x00000000u);
      }
    } else {
      float wVal = W[gid];
      float didjProd = (di * dj);
      float sqrtDiDj = sqrt(didjProd);
      float normalized = (wVal / sqrtDiDj);
      if (isDiag) {
        float lSymVal = (__uint_as_float(0x3f800000u) - normalized);
        L_sym[gid] = lSymVal;
      } else {
        float lSymVal = (__uint_as_float(0x00000000u) - normalized);
        L_sym[gid] = lSymVal;
      }
    }
  }
}
