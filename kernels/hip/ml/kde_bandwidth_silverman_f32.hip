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
extern "C" __global__ void navatala_ml_kde_bandwidth_silverman_f32(const float* stddev, const unsigned int* n, float* bandwidth) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if (gid == 0u) {
    float sigma = stddev[0];
    unsigned int count = n[0];
    float countF = ((float)(count));
    float logN = log(countF);
    float negLogN = (__uint_as_float(0xbe4ccccdu) * logN);
    float nPowNeg02 = exp(negLogN);
    float h = (__uint_as_float(0x3f87ae14u) * (sigma * nPowNeg02));
    bandwidth[0] = h;
  }
}
