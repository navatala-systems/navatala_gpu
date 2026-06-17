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
__kernel void navatala_ml_update_centroids_f64(__global const double* clusterSums, __global const uint* clusterCounts, __global const double* oldCentroids, __global const uint* k, __global const uint* d, __global double* centroids) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint kVal = k[0];
  uint dVal = d[0];
  uint totalElems = (kVal * dVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint clusterIdx = (gid / dVal);
    uint count = clusterCounts[clusterIdx];
    double sumVal = clusterSums[gid];
    double oldVal = oldCentroids[gid];
    bool isEmpty = (count == (uint)(0u));
    double countF = ((double)(count));
    double newVal = (sumVal / countF);
    double result = ((isEmpty) ? (oldVal) : (newVal));
    centroids[gid] = result;
  }
}
