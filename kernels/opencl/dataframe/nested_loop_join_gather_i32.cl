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

__kernel void navatala_dataframe_nested_loop_join_gather_i32(__global const int* leftData, __global const int* rightData, __global const int* leftIndices, __global const int* rightIndices, __global const uint* count, __global int* leftOutput, __global int* rightOutput) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    int leftIdx = leftIndices[gid];
    int rightIdx = rightIndices[gid];
    uint leftIdxU = ((uint)(leftIdx));
    uint rightIdxU = ((uint)(rightIdx));
    int leftVal = leftData[leftIdxU];
    int rightVal = rightData[rightIdxU];
    leftOutput[gid] = leftVal;
    rightOutput[gid] = rightVal;
  }
}
