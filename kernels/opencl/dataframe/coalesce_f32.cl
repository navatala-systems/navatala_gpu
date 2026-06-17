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

__kernel void navatala_dataframe_coalesce_f32(__global const float* col1, __global const uint* col1Valid, __global const float* col2, __global const uint* col2Valid, __global const uint* count, __global float* _output, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint v1Word = col1Valid[wordIdx];
    uint v1Bit = ((v1Word >> bitIdx) & (uint)(1u));
    bool col1IsValid = (v1Bit == (uint)(1u));
    uint v2Word = col2Valid[wordIdx];
    uint v2Bit = ((v2Word >> bitIdx) & (uint)(1u));
    bool col2IsValid = (v2Bit == (uint)(1u));
    float val1 = col1[gid];
    float val2 = col2[gid];
    float outVal = ((col1IsValid) ? (val1) : (val2));
    _output[gid] = outVal;
    bool eitherValid = (col1IsValid || col2IsValid);
    uint outBit = ((eitherValid) ? (((uint)(1u) << bitIdx)) : ((uint)(0u)));
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
