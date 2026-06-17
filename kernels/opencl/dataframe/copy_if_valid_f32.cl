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

__kernel void navatala_dataframe_copy_if_valid_f32(__global const float* src, __global const uint* srcValid, __global const uint* dstValid, __global const uint* count, __global float* dst, __global uint* outValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint srcValidWord = srcValid[wordIdx];
    uint srcValidBit = ((srcValidWord >> bitIdx) & (uint)(1u));
    bool srcIsValid = (srcValidBit == (uint)(1u));
    uint dstValidWord = dstValid[wordIdx];
    uint dstValidBit = ((dstValidWord >> bitIdx) & (uint)(1u));
    bool dstIsValid = (dstValidBit == (uint)(1u));
    bool bothValid = (srcIsValid && dstIsValid);
    if (bothValid) {
      float val = src[gid];
      dst[gid] = val;
      uint outBit = ((uint)(1u) << bitIdx);
      atomic_or(&outValid[wordIdx], outBit);
    }
  }
}
