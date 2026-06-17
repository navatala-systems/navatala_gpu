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
extern "C" __global__ void navatala_graph_compute_degree_matrix_f64(const double* W, const unsigned int* n, double* D) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int wgid = ((unsigned int)((int)(blockIdx.x)));
  __shared__ double sdata[256];
  unsigned int nVal = n[0];
  unsigned int rowIdx = wgid;
  bool rowInBounds = (rowIdx < nVal);
  if (rowInBounds) {
    unsigned int rowBase = (rowIdx * nVal);
    bool colInBounds = (lid < nVal);
    unsigned int elemIdx = (rowBase + lid);
    if (colInBounds) {
      double wVal = W[elemIdx];
      sdata[lid] = wVal;
    } else {
      sdata[lid] = __longlong_as_double(0x0000000000000000ull);
    }
    __syncthreads();
    unsigned int degF64RedStride = 128u;
    for (int degF64RedStep = 0; degF64RedStep < (int)(8); ++degF64RedStep) {
      unsigned int degF64Stride = degF64RedStride;
      if ((lid < degF64Stride)) {
        double other = sdata[(lid + degF64Stride)];
        double mine = sdata[lid];
        double sum = (mine + other);
        sdata[lid] = sum;
      }
      unsigned int degF64StrideToHalve = degF64RedStride;
      unsigned int degF64NextStride = (degF64StrideToHalve >> 1u);
      degF64RedStride = degF64NextStride;
      __syncthreads();
    }
    if ((lid == 0u)) {
      double degreeVal = sdata[0];
      D[rowIdx] = degreeVal;
    }
  }
}
