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
__kernel void navatala_ml_compute_inertia_f64(__global const double* data, __global const double* centroids, __global const uint* assignments, __global const uint* n, __global const uint* k, __global const uint* d, __global double* inertia) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata[256];
  uint nVal = n[0];
  uint dVal = d[0];
  bool inBounds = (gid < nVal);
  double pointDistAccum = as_double(0x0000000000000000ul);
  double distSumAccum = as_double(0x0000000000000000ul);
  if (inBounds) {
    uint cluster = assignments[gid];
    uint dataBase = (gid * dVal);
    uint centBase = (cluster * dVal);
    distSumAccum = as_double(0x0000000000000000ul);
    for (int d = 0; d < (int)(dVal); ++d) {
      uint dU32 = ((uint)(d));
      uint dataIdx = (dataBase + dU32);
      uint centIdx = (centBase + dU32);
      double dataVal = data[dataIdx];
      double centVal = centroids[centIdx];
      double diff = (dataVal - centVal);
      double diffSq = (diff * diff);
      double currDistSum = distSumAccum;
      double newDistSum = (currDistSum + diffSq);
      distSumAccum = newDistSum;
    }
    double distSum = distSumAccum;
    pointDistAccum = distSum;
  }
  double pointDist = pointDistAccum;
  sdata[lid] = pointDist;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint inertiaF64RedStride = (uint)(128u);
  for (int inertiaF64RedStep = 0; inertiaF64RedStep < (int)(8); ++inertiaF64RedStep) {
    uint inertiaF64Stride = inertiaF64RedStride;
    if ((lid < inertiaF64Stride)) {
      double other = sdata[(lid + inertiaF64Stride)];
      double mine = sdata[lid];
      double sum = (mine + other);
      sdata[lid] = sum;
    }
    uint inertiaF64StrideToHalve = inertiaF64RedStride;
    uint inertiaF64NextStride = (inertiaF64StrideToHalve >> (uint)(1u));
    inertiaF64RedStride = inertiaF64NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    inertia[0] = sdata[0];
  }
}
