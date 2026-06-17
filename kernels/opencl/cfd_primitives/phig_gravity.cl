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

__kernel void navatala_cfd_primitives_phig_gravity(__global const float* ghf, __global const float* snGradRho, __global const float* rAUf, __global const float* magSf, __global const uint* params, __global float* outPhig) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= ((int)(params[0])))) {
    return;
  } else {
    outPhig[(int)(get_global_id(0))] = (-((ghf[(int)(get_global_id(0))] * snGradRho[(int)(get_global_id(0))]) * (rAUf[(int)(get_global_id(0))] * magSf[(int)(get_global_id(0))])));
  }
}
