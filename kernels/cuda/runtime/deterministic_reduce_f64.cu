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
extern "C" __global__ void navatala_runtime_deterministic_reduce_f64(const double* _input, const unsigned int* count, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    double val = _input[gid];
    sdata[lid] = val;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int f64RedStride = 128u;
  for (int f64RedStep = 0; f64RedStep < (int)(8); ++f64RedStep) {
    unsigned int f64Stride = f64RedStride;
    if (lid < f64Stride) {
      double other = sdata[(lid + f64Stride)];
      double mine = sdata[lid];
      double sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int f64StrideToHalve = f64RedStride;
    unsigned int f64NextStride = (f64StrideToHalve >> 1u);
    f64RedStride = f64NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    double totalSum = sdata[0];
    result[0] = totalSum;
  }
}
