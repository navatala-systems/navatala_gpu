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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void navatala_transformer_multi_tensor_clip_grad_f16(__global const float* totalNorm, __global const float* maxNorm, __global const uint* totalElements, __global half* grad) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((((uint)((int)(get_group_id(0)))) * (uint)(256u)) + ((uint)((int)(get_local_id(0)))));
  uint total = totalElements[(uint)(0u)];
  float tNorm = totalNorm[(uint)(0u)];
  float mNorm = maxNorm[(uint)(0u)];
  float epsilon = as_float(0x358637bdu);
  float safeTNorm = (tNorm + epsilon);
  float ratio = (mNorm / safeTNorm);
  bool needsClip = (tNorm > mNorm);
  float clipCoef = ((needsClip) ? (ratio) : (as_float(0x3f800000u)));
  bool valid = (gid < total);
  if (valid) {
    half gF16 = grad[gid];
    float g = ((float)(gF16));
    float clippedF32 = (g * clipCoef);
    half clipped = ((half)(clippedF32));
    grad[gid] = clipped;
  }
}
