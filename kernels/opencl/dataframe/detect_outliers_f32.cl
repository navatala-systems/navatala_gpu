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

__kernel void navatala_dataframe_detect_outliers_f32(__global const float* data, __global const uint* n, __global const float* median, __global const float* mad, __global const float* threshold, __global uint* isOutlier) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint len = n[0];
  bool inBounds = (gid < len);
  if (inBounds) {
    float med = median[0];
    float m = mad[0];
    float thresh = threshold[0];
    float xi = data[gid];
    float dev = fabs((xi - med));
    float cutoff = (thresh * m);
    bool isOut = (dev > cutoff);
    uint flag = ((isOut) ? ((uint)(1u)) : ((uint)(0u)));
    isOutlier[gid] = flag;
  }
}
