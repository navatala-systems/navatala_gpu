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

__kernel void navatala_cfd_scatter_add_atomic(__global const int* values, __global const int* indices, __global const int* numEdges, __global const int* featureDim, __global int* dst) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nEdges = numEdges[0];
  if ((gid < nEdges)) {
    int dstIdx = indices[gid];
    int fd = featureDim[0];
    for (int f = 0; f < (int)(fd); ++f) {
      int valFlat = ((gid * fd) + f);
      int dstFlat = ((dstIdx * fd) + f);
      int v = values[valFlat];
      int _ignored = atomic_add(&(dst[dstFlat]), v);
    }
  }
}
