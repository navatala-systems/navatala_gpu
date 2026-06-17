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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_cfd_radius_count(const float* query, const float* ref, const int* numQuery, const int* numRef, const float* radiusIn, int* counts) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int qid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int nq = numQuery[0];
  int nr = numRef[0];
  if ((qid < nq)) {
    float qx = query[((qid)*3 + 0)];
    float qy = query[((qid)*3 + 1)];
    float qz = query[((qid)*3 + 2)];
    float rIn = radiusIn[0];
    float rCmp = rIn;
    int cnt = 0;
    for (int rid = 0; rid < (int)(nr); ++rid) {
      float rx = ref[((rid)*3 + 0)];
      float ry = ref[((rid)*3 + 1)];
      float rz = ref[((rid)*3 + 2)];
      float dx = (qx - rx);
      float dy = (qy - ry);
      float dz = (qz - rz);
      float d2 = (((dx * dx) + (dy * dy)) + (dz * dz));
      if ((d2 <= rCmp)) {
        cnt = (cnt + 1);
      }
    }
    counts[qid] = cnt;
  }
}
