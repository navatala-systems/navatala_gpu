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
extern "C" __global__ void navatala_graph_normalize_eigenvector_rows_f64(const double* V, const unsigned int* n, const unsigned int* k, double* V_norm) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ double sdata[256];
  unsigned int nVal = n[0];
  unsigned int kVal = k[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int rowBase = (rowIdx * kVal);
    bool colInBounds = (lid < kVal);
    unsigned int elemIdx = (rowBase + lid);
    if (colInBounds) {
      double vVal = V[elemIdx];
      double vSq = (vVal * vVal);
      sdata[lid] = vSq;
    } else {
      sdata[lid] = __longlong_as_double(0x0000000000000000ull);
    }
    __syncthreads();
    unsigned int normF64RedStride = 128u;
    for (int normF64RedStep = 0; normF64RedStep < (int)(8); ++normF64RedStep) {
      unsigned int normF64Stride = normF64RedStride;
      if ((lid < normF64Stride)) {
        double other = sdata[(lid + normF64Stride)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int normF64StrideToHalve = normF64RedStride;
      unsigned int normF64NextStride = (normF64StrideToHalve >> 1u);
      normF64RedStride = normF64NextStride;
      __syncthreads();
    }
    if ((lid == 0u)) {
      double sumSq = sdata[0];
      double normVal = sqrt(sumSq);
      sdata[0] = normVal;
    }
    __syncthreads();
    if (colInBounds) {
      double norm = sdata[0];
      double vVal2 = V[elemIdx];
      double eps = __longlong_as_double(0x3cd203af9ee75616ull);
      bool normZero = (norm < eps);
      if (normZero) {
        V_norm[elemIdx] = __longlong_as_double(0x0000000000000000ull);
      } else {
        double normalized = (vVal2 / norm);
        V_norm[elemIdx] = normalized;
      }
    }
  }
}
