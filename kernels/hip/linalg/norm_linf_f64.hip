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
extern "C" __global__ void navatala_linalg_norm_linf_f64(const double* _input, const unsigned int* count, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  double val = abs(_input[gid]);
  if (inBounds) {
    sdata[lid] = val;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int linfF64RedStride = 128u;
  for (int linfF64RedStep = 0; linfF64RedStep < (int)(8); ++linfF64RedStep) {
    unsigned int linfF64Stride = linfF64RedStride;
    if ((lid < linfF64Stride)) {
      double other = sdata[(lid + linfF64Stride)];
      double mine = sdata[lid];
      bool mineGtOther = (mine > other);
      double maxVal = ((mineGtOther) ? (mine) : (other));
      sdata[lid] = maxVal;
    }
    unsigned int linfF64StrideToHalve = linfF64RedStride;
    unsigned int linfF64NextStride = (linfF64StrideToHalve >> 1u);
    linfF64RedStride = linfF64NextStride;
    __syncthreads();
  }
  if ((lid == 0u)) {
    result[0] = sdata[0];
  }
}
