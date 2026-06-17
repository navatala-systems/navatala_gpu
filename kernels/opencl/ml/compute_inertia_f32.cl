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

__kernel void navatala_ml_compute_inertia_f32(__global const float* data, __global const float* centroids, __global const uint* assignments, __global const uint* n, __global const uint* k, __global const uint* d, __global float* inertia) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint nVal = n[0];
  uint dVal = d[0];
  bool inBounds = (gid < nVal);
  float pointDistAccum = as_float(0x00000000u);
  float distSumAccum = as_float(0x00000000u);
  if (inBounds) {
    uint cluster = assignments[gid];
    uint dataBase = (gid * dVal);
    uint centBase = (cluster * dVal);
    distSumAccum = as_float(0x00000000u);
    for (int d = 0; d < (int)(dVal); ++d) {
      uint dU32 = ((uint)(d));
      uint dataIdx = (dataBase + dU32);
      uint centIdx = (centBase + dU32);
      float dataVal = data[dataIdx];
      float centVal = centroids[centIdx];
      float diff = (dataVal - centVal);
      float diffSq = (diff * diff);
      float currDistSum = distSumAccum;
      float newDistSum = (currDistSum + diffSq);
      distSumAccum = newDistSum;
    }
    float distSum = distSumAccum;
    pointDistAccum = distSum;
  }
  float pointDist = pointDistAccum;
  sdata[lid] = pointDist;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint inertiaF32RedStride = (uint)(128u);
  for (int inertiaF32RedStep = 0; inertiaF32RedStep < (int)(8); ++inertiaF32RedStep) {
    uint inertiaF32Stride = inertiaF32RedStride;
    if ((lid < inertiaF32Stride)) {
      float other = sdata[(lid + inertiaF32Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint inertiaF32StrideToHalve = inertiaF32RedStride;
    uint inertiaF32NextStride = (inertiaF32StrideToHalve >> (uint)(1u));
    inertiaF32RedStride = inertiaF32NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    inertia[0] = sdata[0];
  }
}
