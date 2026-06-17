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
extern "C" __global__ void navatala_dataframe_uniform_lcg_f64(unsigned long long* state, const unsigned int* count, double* _output) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0u];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    unsigned long long s = state[gid];
    unsigned long long aU64 = ((unsigned long long)(1103515245u));
    unsigned long long cU64 = ((unsigned long long)(12345u));
    unsigned long long prod = (aU64 * s);
    unsigned long long sum = (prod + cU64);
    unsigned long long newState = (sum & ((unsigned long long)(2147483647u)));
    state[gid] = newState;
    double stateF64 = ((double)(newState));
    double randVal = (stateF64 / __longlong_as_double(0x41e0000000000000ull));
    _output[gid] = randVal;
  }
}
