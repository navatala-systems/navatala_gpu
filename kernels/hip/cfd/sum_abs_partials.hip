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
extern "C" __global__ void navatala_cfd_sum_abs_partials(const float* a, const int* counts, float* outPartials) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  __shared__ double tmp[256];
  float v = __uint_as_float(0x00000000u);
  if ((((int)((int)(blockIdx.x * blockDim.x + threadIdx.x))) < counts[0])) {
    float ai = a[((int)((int)(blockIdx.x * blockDim.x + threadIdx.x)))];
    if ((ai >= __uint_as_float(0x00000000u))) {
      v = ai;
    } else {
      v = (__uint_as_float(0x00000000u) - ai);
    }
  }
  tmp[((int)((int)(threadIdx.x)))] = v;
  __syncthreads();
  if ((((int)((int)(threadIdx.x))) < 128)) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 128)]);
  }
  __syncthreads();
  if ((((int)((int)(threadIdx.x))) < 64)) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 64)]);
  }
  __syncthreads();
  if ((((int)((int)(threadIdx.x))) < 32)) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 32)]);
  }
  __syncthreads();
  if ((((int)((int)(threadIdx.x))) < 16)) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 16)]);
  }
  __syncthreads();
  if ((((int)((int)(threadIdx.x))) < 8)) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 8)]);
  }
  __syncthreads();
  if ((((int)((int)(threadIdx.x))) < 4)) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 4)]);
  }
  __syncthreads();
  if ((((int)((int)(threadIdx.x))) < 2)) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 2)]);
  }
  __syncthreads();
  if ((((int)((int)(threadIdx.x))) < 1)) {
    tmp[((int)((int)(threadIdx.x)))] = (tmp[((int)((int)(threadIdx.x)))] + tmp[(((int)((int)(threadIdx.x))) + 1)]);
  }
  __syncthreads();
  if ((((int)((int)(threadIdx.x))) == 0)) {
    outPartials[((int)((int)(blockIdx.x)))] = tmp[0];
  }
}
