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

__kernel void navatala_dataframe_subtract_timestamps(__global const long* timestamps1, __global const long* timestamps2, __global const uint* validity1, __global const uint* validity2, __global const uint* count, __global long* _output, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord1 = validity1[wordIdx];
    uint validBit1 = ((validWord1 >> bitIdx) & (uint)(1u));
    uint validWord2 = validity2[wordIdx];
    uint validBit2 = ((validWord2 >> bitIdx) & (uint)(1u));
    uint bothValid = (validBit1 & validBit2);
    bool isValid = (bothValid == (uint)(1u));
    long ts1 = timestamps1[gid];
    long ts2 = timestamps2[gid];
    long diff = (ts2 - ts1);
    long outVal = ((isValid) ? (diff) : (0));
    _output[gid] = outVal;
    uint outBit = (bothValid << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
