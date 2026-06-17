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

__kernel void navatala_dataframe_column_arg_max_f32(__global const float* data, __global const uint* validity, __global const uint* count, __global int* maxValBits, __global uint* maxIdx) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint validWord = validity[wordIdx];
    uint validBit = ((validWord >> bitIdx) & (uint)(1u));
    bool isValid = (validBit == (uint)(1u));
    if (isValid) {
      float val = data[gid];
      int valBits = ((int)(val));
      atomic_max(&maxValBits[(uint)(0u)], valBits);
      int currMaxBits = maxValBits[(uint)(0u)];
      bool isMax = (valBits == currMaxBits);
      if (isMax) {
        atomic_min(&maxIdx[(uint)(0u)], gid);
      }
    }
  }
}
