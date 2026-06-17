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
__kernel void navatala_linalg_strided_slice_f64(__global const double* A, __global const uint* start, __global const uint* stride, __global const uint* count, __global double* _out) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint idx = gid;
  uint s = start[(uint)(0u)];
  uint st = stride[(uint)(0u)];
  uint n = count[(uint)(0u)];
  if ((idx < n)) {
    uint srcIdx = (s + (idx * st));
    double val = A[srcIdx];
    _out[idx] = val;
  }
}
