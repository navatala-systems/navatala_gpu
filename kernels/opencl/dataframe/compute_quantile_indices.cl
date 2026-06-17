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

__kernel void navatala_dataframe_compute_quantile_indices(__global const uint* quantile, __global const uint* count, __global uint* floorIdx, __global uint* ceilIdx, __global uint* fraction) {
  int gid0 = (int)get_global_id(0);
  uint q = quantile[(uint)(0u)];
  uint n = count[(uint)(0u)];
  uint nMinus1 = (n - (uint)(1u));
  uint posScaled = (q * nMinus1);
  uint floor = (posScaled / (uint)(1000u));
  uint ceilRaw = (floor + (uint)(1u));
  uint ceil = (((ceilRaw < n)) ? (ceilRaw) : (nMinus1));
  uint frac = (posScaled % (uint)(1000u));
  floorIdx[(uint)(0u)] = floor;
  ceilIdx[(uint)(0u)] = ceil;
  fraction[(uint)(0u)] = frac;
}
