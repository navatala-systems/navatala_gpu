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
extern "C" __global__ void navatala_dataframe_tdigest_quantile_f32(const float* p, const float* centroidMeans, const float* centroidWeights, const unsigned int* centroidCount, const float* totalWeight, const float* minVal, const float* maxVal, float* quantile) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  if ((gid == 0u)) {
    float pVal = p[0];
    unsigned int count = centroidCount[0];
    float minV = minVal[0];
    float maxV = maxVal[0];
    float range = (maxV - minV);
    float result = (minV + (pVal * range));
    bool isEmpty = (count == 0u);
    float finalResult = ((isEmpty) ? (__uint_as_float(0x00000000u)) : (result));
    quantile[0] = finalResult;
  }
}
