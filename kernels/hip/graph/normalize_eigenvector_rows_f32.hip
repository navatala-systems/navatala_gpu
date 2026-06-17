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
extern "C" __global__ void navatala_graph_normalize_eigenvector_rows_f32(const float* V, const unsigned int* n, const unsigned int* k, float* V_norm) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ float sdata[256];
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int rowBase = (rowIdx * kVal);
    bool colInBounds = (lid < kVal);
    unsigned int elemIdx = (rowBase + lid);
    if (colInBounds) {
      float vVal = V[elemIdx];
      float vSq = (vVal * vVal);
      sdata[lid] = vSq;
    } else {
      sdata[lid] = __uint_as_float(0x00000000u);
    }
    __syncthreads();
    unsigned int normF32RedStride = 128u;
    for (int normF32RedStep = 0; normF32RedStep < (int)(8); ++normF32RedStep) {
      unsigned int normF32Stride = normF32RedStride;
      if ((lid < normF32Stride)) {
        float other = sdata[(lid + normF32Stride)];
        float mine = sdata[lid];
        float sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int normF32StrideToHalve = normF32RedStride;
      unsigned int normF32NextStride = (normF32StrideToHalve >> 1u);
      normF32RedStride = normF32NextStride;
      __syncthreads();
    }
    if ((lid == 0u)) {
      float sumSq = sdata[0];
      float normVal = sqrt(sumSq);
      sdata[0] = normVal;
    }
    __syncthreads();
    if (colInBounds) {
      float norm = sdata[0];
      float vVal2 = V[elemIdx];
      float eps = __uint_as_float(0x2edbe6ffu);
      bool normZero = (norm < eps);
      if (normZero) {
        V_norm[elemIdx] = __uint_as_float(0x00000000u);
      } else {
        float normalized = (vVal2 / norm);
        V_norm[elemIdx] = normalized;
      }
    }
  }
}
