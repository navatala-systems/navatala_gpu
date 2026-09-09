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

__kernel void navatala_samples_det_prepare_scatter_0(__global const uint* dstIdx, __global const float* messages, __global const int* detN, __global uint* keys_scatter_0, __global float* vals_scatter_0) {
  int gid0 = (int)get_global_id(0);
  if ((int)(get_global_id(0)) < detN[0]) {
    keys_scatter_0[(int)(get_global_id(0))] = dstIdx[(int)(get_global_id(0))];
    vals_scatter_0[(int)(get_global_id(0))] = messages[(int)(get_global_id(0))];
  }
}
