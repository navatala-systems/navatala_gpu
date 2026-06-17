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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_cfd_radius_count(device const float* query [[buffer(0)]], device const float* ref [[buffer(1)]], device const int* numQuery [[buffer(2)]], device const int* numRef [[buffer(3)]], device const float* radiusIn [[buffer(4)]], device int* counts [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  int qid = int(__gid.x);
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
