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

__kernel void navatala_vector_search_decode_p_q_vectors_f32(__global const uchar* codes, __global const float* codebook, __global const uint* n_vectors, __global const uint* sub_dim, __global float* _output) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nv = n_vectors[0];
  uint sd = sub_dim[0];
  uint total = (nv * sd);
  if ((gid < total)) {
    uint vid = (gid / sd);
    uint k = (gid % sd);
    uchar code = codes[vid];
    uint cw = ((uint)(code));
    uint cb_idx = ((cw * sd) + k);
    float val = codebook[cb_idx];
    _output[gid] = val;
  }
}
