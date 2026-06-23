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

__kernel void navatala_cfd_primitives_face_scalar_product(__global const float* aAllFaces, __global const float* bAllFaces, __global const int* params, __global float* outProd) {
  int gid0 = (int)get_global_id(0);
  if (((int)((int)(get_global_id(0)))) >= params[0]) {
    return;
  } else {
    float a = aAllFaces[((int)((int)(get_global_id(0))))];
    float b = bAllFaces[((int)((int)(get_global_id(0))))];
    outProd[((int)((int)(get_global_id(0))))] = (a * b);
  }
}
