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

__kernel void navatala_dataframe_bitwise_or_i32(__global const int* left, __global const uint* leftValid, __global const int* right, __global const uint* rightValid, __global const uint* count, __global int* _output, __global uint* outputValid) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int lval = left[gid];
    int rval = right[gid];
    int result = (lval | rval);
    _output[gid] = result;
    uint wordIdx = (gid / (uint)(32u));
    uint bitIdx = (gid % (uint)(32u));
    uint leftValidWord = leftValid[wordIdx];
    uint leftValidBit = ((leftValidWord >> bitIdx) & (uint)(1u));
    uint rightValidWord = rightValid[wordIdx];
    uint rightValidBit = ((rightValidWord >> bitIdx) & (uint)(1u));
    uint outValidBit = (leftValidBit & rightValidBit);
    uint outBit = (outValidBit << bitIdx);
    atomic_or(&outputValid[wordIdx], outBit);
  }
}
