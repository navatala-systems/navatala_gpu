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

__kernel void navatala_ml_count_clusters_f32(__global const int* labels, __global const uint* nPoints, __global int* numClusters) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local int sdata[256];
  uint n = nPoints[0];
  bool inBounds = (gid < n);
  int myLabel = ((inBounds) ? (labels[gid]) : (-1));
  sdata[lid] = myLabel;
  barrier(CLK_LOCAL_MEM_FENCE);
  uint countClstF32RedStride = (uint)(128u);
  for (int countClstF32RedStep = 0; countClstF32RedStep < (int)(8); ++countClstF32RedStep) {
    uint countClstF32Stride = countClstF32RedStride;
    if ((lid < countClstF32Stride)) {
      uint otherIdx = (lid + countClstF32Stride);
      int other = sdata[otherIdx];
      int mine = sdata[lid];
      bool isGreater = (other > mine);
      int maxVal = ((isGreater) ? (other) : (mine));
      sdata[lid] = maxVal;
    }
    uint countClstF32StrideToHalve = countClstF32RedStride;
    uint countClstF32NextStride = (countClstF32StrideToHalve >> (uint)(1u));
    countClstF32RedStride = countClstF32NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == (uint)(0u))) {
    int maxLabel = sdata[0];
    int count = (maxLabel + 1);
    numClusters[0] = count;
  }
}
