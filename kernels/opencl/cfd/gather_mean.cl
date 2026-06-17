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

__kernel void navatala_cfd_gather_mean(__global const float* features, __global const int* neighbors, __global const int* neighborCounts, __global const int* numNodes, __global const int* featureDim, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int nNodes = numNodes[0];
  if ((gid < nNodes)) {
    int cnt = neighborCounts[gid];
    int fd = featureDim[0];
    for (int f = 0; f < (int)(fd); ++f) {
      float acc = as_float(0x00000000u);
      for (int j = 0; j < (int)(cnt); ++j) {
        int nbrFlat = ((gid * 8) + j);
        int nbr = neighbors[nbrFlat];
        int featFlat = ((nbr * fd) + f);
        float v = features[featFlat];
        acc = (acc + v);
      }
      float cntF = ((float)(cnt));
      float mean = (((cnt > 0)) ? ((acc / cntF)) : (as_float(0x00000000u)));
      int outFlat = ((gid * fd) + f);
      _out[outFlat] = mean;
    }
  }
}
