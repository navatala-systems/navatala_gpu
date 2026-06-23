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

__kernel void navatala_cfd_mg_renumber_propagate(__global int* aggMap, __global const int* counts) {
  int gid0 = (int)get_global_id(0);
  const int nSafeMax = (((int)(counts[0])) > 0 ? ((int)(counts[0])) - 1 : 0);
  const int safeIdx = (gid0 < nSafeMax ? gid0 : nSafeMax);
  if (gid0 >= ((int)(counts[0]))) return;
  if (((int)(get_global_id(0)) >= ((int)(counts[0])))) {
    return;
  } else {
    int root = aggMap[(int)(get_global_id(0))];
    if ((root < 0)) {
      aggMap[(int)(get_global_id(0))] = (-(root + 1));
    } else {
      int rootVal = aggMap[root];
      if ((rootVal < 0)) {
        aggMap[(int)(get_global_id(0))] = (-(rootVal + 1));
      } else {
        aggMap[(int)(get_global_id(0))] = rootVal;
      }
    }
  }
}
