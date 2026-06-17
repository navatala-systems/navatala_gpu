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

__kernel void navatala_dataframe_split_copy_validity(__global const uint* srcValidity, __global const uint* srcOffset, __global const uint* count, __global uint* dstValidity) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint offset = srcOffset[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint srcRowIdx = (offset + gid);
    uint srcWordIdx = (srcRowIdx / (uint)(32u));
    uint srcBitIdx = (srcRowIdx % (uint)(32u));
    uint srcWord = srcValidity[srcWordIdx];
    uint srcBit = ((srcWord >> srcBitIdx) & (uint)(1u));
    bool isValid = (srcBit == (uint)(1u));
    uint dstWordIdx = (gid / (uint)(32u));
    uint dstBitIdx = (gid % (uint)(32u));
    uint dstBit = ((isValid) ? (((uint)(1u) << dstBitIdx)) : ((uint)(0u)));
    atomic_or(&dstValidity[dstWordIdx], dstBit);
  }
}
