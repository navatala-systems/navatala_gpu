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

__kernel void navatala_dataframe_median_f32(__global const float* sortedInput, __global const uint* count, __global float* median) {
  int gid0 = (int)get_global_id(0);
  uint gidU32 = ((uint)((int)(get_global_id(0))));
  if (gidU32 == (uint)(0u)) {
    uint n = count[(uint)(0u)];
    uint nMinus1 = (n - (uint)(1u));
    float nMinus1Float = ((float)(nMinus1));
    float idx = (as_float(0x3f000000u) * nMinus1Float);
    float lowerFloat = floor(idx);
    uint lower = ((uint)(lowerFloat));
    uint upperCandidate = (lower + (uint)(1u));
    uint upper = (((upperCandidate < n)) ? (upperCandidate) : (nMinus1));
    float frac = (idx - lowerFloat);
    float valLower = sortedInput[lower];
    float valUpper = sortedInput[upper];
    float oneMinusFrac = (as_float(0x3f800000u) - frac);
    float termLower = (oneMinusFrac * valLower);
    float termUpper = (frac * valUpper);
    float medianVal = (termLower + termUpper);
    median[(uint)(0u)] = medianVal;
  }
}
