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

__kernel void navatala_dataframe_rolling_variance_helper_f32(__global const float* data, __global const uint* validity, __global const uint* windowSize, __global const uint* count, __global long* sumOutput, __global long* sumSqOutput, __global uint* countOutput, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    float val = data[gid];
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    float scaleFactor = as_float(0x47800000u);
    float scaledVal = (val * scaleFactor);
    long scaledInt = ((long)(scaledVal));
    long sumVal = ((isValid) ? (scaledInt) : (0));
    float valSq = (val * val);
    float scaledSq = (valSq * scaleFactor);
    long scaledSqInt = ((long)(scaledSq));
    long sumSqVal = ((isValid) ? (scaledSqInt) : (0));
    uint cntVal = ((isValid) ? ((uint)(1u)) : ((uint)(0u)));
    sumOutput[gid] = sumVal;
    sumSqOutput[gid] = sumSqVal;
    countOutput[gid] = cntVal;
    uint outBit = (validBit << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
