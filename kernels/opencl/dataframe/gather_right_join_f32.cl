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

__kernel void navatala_dataframe_gather_right_join_f32(__global const float* rightData, __global const uint* rightValidity, __global const int* rightIndices, __global const uint* outputCount, __global float* _output, __global uint* outputValidity) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = outputCount[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int rightIdx = rightIndices[gid];
    bool hasMatch = (rightIdx >= 0);
    uint outWordIdx = (gid / (uint)(32u));
    uint outBitIdx = (gid % (uint)(32u));
    if (hasMatch) {
      uint rightIdxU = ((uint)(rightIdx));
      float val = rightData[rightIdxU];
      uint srcWordIdx = (rightIdxU / (uint)(32u));
      uint srcBitIdx = (rightIdxU % (uint)(32u));
      uint srcWord = rightValidity[srcWordIdx];
      uint srcBit = ((srcWord >> srcBitIdx) & (uint)(1u));
      _output[gid] = val;
      uint outBit = (srcBit << outBitIdx);
      atomic_or(&outputValidity[outWordIdx], outBit);
    } else {
      _output[gid] = as_float(0x00000000u);
    }
  }
}
