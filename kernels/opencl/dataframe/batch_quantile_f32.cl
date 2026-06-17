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

__kernel void navatala_dataframe_batch_quantile_f32(__global const float* sortedData, __global const uint* quantiles, __global const uint* dataCount, __global const uint* numQuantiles, __global float* results) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nQ = numQuantiles[(uint)(0u)];
  uint nData = dataCount[(uint)(0u)];
  bool inBounds = (gid < nQ);
  if (inBounds) {
    uint q = quantiles[gid];
    uint nMinus1 = (nData - (uint)(1u));
    uint posScaled = (q * nMinus1);
    uint floor = (posScaled / (uint)(1000u));
    uint ceilRaw = (floor + (uint)(1u));
    uint ceil = (((ceilRaw < nData)) ? (ceilRaw) : (nMinus1));
    uint frac = (posScaled % (uint)(1000u));
    float floorVal = sortedData[floor];
    float ceilVal = sortedData[ceil];
    float diff = (ceilVal - floorVal);
    float fracF = ((float)(frac));
    float scaledDiff = (diff * fracF);
    float interpVal = (scaledDiff / as_float(0x447a0000u));
    float quantileVal = (floorVal + interpVal);
    results[gid] = quantileVal;
  }
}
