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

__kernel void navatala_dataframe_mutual_information_f32(__global const float* hX, __global const float* hY, __global const float* hXY, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gidU32 = ((uint)((int)(get_global_id(0))));
  if ((gidU32 == (uint)(0u))) {
    float hXVal = hX[(uint)(0u)];
    float hYVal = hY[(uint)(0u)];
    float hXYVal = hXY[(uint)(0u)];
    float sum = (hXVal + hYVal);
    float mi = (sum - hXYVal);
    result[(uint)(0u)] = mi;
  }
}
