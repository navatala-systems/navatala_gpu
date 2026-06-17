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

__kernel void navatala_sparse_amg_coarse_size_check_f32(__global const uint* coarseNRows, __global const uint* denseLUThreshold, __global uint* useDenseLU) {
  int gid0 = (int)get_global_id(0);
  uint cn = coarseNRows[0];
  uint th = denseLUThreshold[0];
  uint flag = (((cn < th)) ? ((uint)(1u)) : ((uint)(0u)));
  useDenseLU[0] = flag;
}
