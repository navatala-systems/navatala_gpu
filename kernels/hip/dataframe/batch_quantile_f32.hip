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
extern "C" __global__ void navatala_dataframe_batch_quantile_f32(const float* sortedData, const unsigned int* quantiles, const unsigned int* dataCount, const unsigned int* numQuantiles, float* results) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int nQ = numQuantiles[0u];
  unsigned int nData = dataCount[0u];
  bool inBounds = (gid < nQ);
  if (inBounds) {
    unsigned int q = quantiles[gid];
    unsigned int nMinus1 = (nData - 1u);
    unsigned int posScaled = (q * nMinus1);
    unsigned int floor = (posScaled / 1000u);
    unsigned int ceilRaw = (floor + 1u);
    unsigned int ceil = (((ceilRaw < nData)) ? (ceilRaw) : (nMinus1));
    unsigned int frac = (posScaled % 1000u);
    float floorVal = sortedData[floor];
    float ceilVal = sortedData[ceil];
    float diff = (ceilVal - floorVal);
    float fracF = ((float)(frac));
    float scaledDiff = (diff * fracF);
    float interpVal = (scaledDiff / __uint_as_float(0x447a0000u));
    float quantileVal = (floorVal + interpVal);
    results[gid] = quantileVal;
  }
}
