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
extern "C" __global__ void navatala_dataframe_init_reservoir256_f64(const unsigned int* initialSeed, double* reservoir, unsigned int* count, unsigned int* seed) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  bool inBounds = ((int)(blockIdx.x * blockDim.x + threadIdx.x) < 256);
  if (inBounds) {
    reservoir[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = __longlong_as_double(0x0000000000000000ull);
  }
  if ((int)(blockIdx.x * blockDim.x + threadIdx.x) == 0) {
    unsigned int initSeed = initialSeed[0];
    count[0] = 0u;
    seed[0] = initSeed;
  }
}
