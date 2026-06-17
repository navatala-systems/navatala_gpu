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

__kernel void navatala_dataframe_select_median_f32(__global const float* sortedData, __global const uint* floorIdx, __global const uint* ceilIdx, __global const uint* isEven, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint floor = floorIdx[(uint)(0u)];
  uint ceil = ceilIdx[(uint)(0u)];
  uint even = isEven[(uint)(0u)];
  bool isEvenBool = (even == (uint)(1u));
  float floorVal = sortedData[floor];
  float ceilVal = sortedData[ceil];
  float sum = (floorVal + ceilVal);
  float avg = (sum / as_float(0x40000000u));
  float median = ((isEvenBool) ? (avg) : (floorVal));
  result[(uint)(0u)] = median;
}
