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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_graph_unpack_f64x6(__global const int* indices, __global const double* src, __global const int* count, __global const int* n, __global double* dst) {
  int gid0 = (int)get_global_id(0);
  int i = ((int)((int)(get_global_id(0))));
  if (i < count[0]) {
    for (int c = 0; c < (int)(6); ++c) {
      int srcLin = ((c * count[0]) + i);
      int dstLin = ((c * n[0]) + indices[((int)((int)(get_global_id(0))))]);
      dst[dstLin] = src[srcLin];
    }
  }
}
