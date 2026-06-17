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

__kernel void navatala_dataframe_rolling_sum_f32(__global const float* data, __global const uint* validity, __global const uint* windowSize, __global const uint* minPeriods, __global const uint* count, __global float* _output, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint w = windowSize[(uint)(0u)];
  uint minP = minPeriods[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int windowStartRaw = (((int)(gid)) - (((int)(w)) - 1));
    int windowStartClamped = (((windowStartRaw < 0)) ? (0) : (windowStartRaw));
    uint windowStart = ((uint)(windowStartClamped));
    float val = data[gid];
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    float outVal = ((isValid) ? (val) : (as_float(0x00000000u)));
    _output[gid] = outVal;
    uint outBit = (validBit << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
