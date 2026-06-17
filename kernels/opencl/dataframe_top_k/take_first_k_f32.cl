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

__kernel void navatala_dataframe_top_k_take_first_k_f32(__global const float* valuesIn, __global const uint* indicesIn, __global const uint* k, __global float* valuesOut, __global uint* indicesOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint kVal = k[0];
  bool inBounds = (gid < kVal);
  if (inBounds) {
    float val = valuesIn[gid];
    uint idx = indicesIn[gid];
    valuesOut[gid] = val;
    indicesOut[gid] = idx;
  }
}
