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

__kernel void navatala_dataframe_gather_left_join_i32(__global const int* leftData, __global const uint* leftValidity, __global const int* leftIndices, __global const uint* outputCount, __global int* _output, __global uint* outputValidity) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = outputCount[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int leftIdx = leftIndices[gid];
    bool hasMatch = (leftIdx >= 0);
    uint outWordIdx = (gid / (uint)(32u));
    uint outBitIdx = (gid % (uint)(32u));
    if (hasMatch) {
      uint leftIdxU = ((uint)(leftIdx));
      int val = leftData[leftIdxU];
      uint srcWordIdx = (leftIdxU / (uint)(32u));
      uint srcBitIdx = (leftIdxU % (uint)(32u));
      uint srcWord = leftValidity[srcWordIdx];
      uint srcBit = ((srcWord >> srcBitIdx) & (uint)(1u));
      _output[gid] = val;
      uint outBit = (srcBit << outBitIdx);
      atomic_or(&outputValidity[outWordIdx], outBit);
    } else {
      _output[gid] = 0;
    }
  }
}
