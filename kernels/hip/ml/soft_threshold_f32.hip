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
extern "C" __global__ void navatala_ml_soft_threshold_f32(const float* value, const float* threshold, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  bool inBounds = (gid == 0u);
  if (inBounds) {
    float v = value[0];
    float t = threshold[0];
    float absV = abs(v);
    float diff = (absV - t);
    float maxDiff = (((diff > __uint_as_float(0x00000000u))) ? (diff) : (__uint_as_float(0x00000000u)));
    bool positive = (v > __uint_as_float(0x00000000u));
    bool negative = (v < __uint_as_float(0x00000000u));
    float signVal = ((positive) ? (__uint_as_float(0x3f800000u)) : (((negative) ? (__uint_as_float(0xbf800000u)) : (__uint_as_float(0x00000000u)))));
    float res = (signVal * maxDiff);
    result[0] = res;
  }
}
