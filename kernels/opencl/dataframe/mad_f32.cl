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

__kernel void navatala_dataframe_mad_f32(__global const float* sortedData, __global const uint* n, __global const float* median, __global float* mad) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  if (gid == (uint)(0u)) {
    uint len = n[0];
    float med = median[0];
    uint q1Idx = (len / (uint)(4u));
    uint q3Idx = (len - q1Idx);
    float q1Val = sortedData[q1Idx];
    float q3Val = sortedData[q3Idx];
    float dev1 = fabs((q1Val - med));
    float dev3 = fabs((q3Val - med));
    float madVal = ((dev1 + dev3) / as_float(0x40000000u));
    mad[0] = madVal;
  }
}
