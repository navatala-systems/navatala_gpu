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

__kernel void navatala_dataframe_writeback_sum_f32(__global const uint* keys, __global const float* vals, __global const int* count, __global float* dst) {
  int gid0 = (int)get_global_id(0);
  int j = ((int)((int)(get_global_id(0))));
  if ((j < count[(uint)(0u)])) {
    uint key = keys[j];
    float val = vals[j];
    float oldVal = dst[key];
    dst[key] = (oldVal + val);
  }
}
