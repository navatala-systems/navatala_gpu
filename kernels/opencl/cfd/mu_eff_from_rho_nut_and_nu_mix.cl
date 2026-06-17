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

__kernel void navatala_cfd_mu_eff_from_rho_nut_and_nu_mix(__global const float* rho, __global const float* nut, __global const float* nuMix, __global const uint* params, __global float* outMu) {
  int gid0 = (int)get_global_id(0);
  if (((int)(get_global_id(0)) >= ((int)(params[0])))) {
    return;
  } else {
    outMu[(int)(get_global_id(0))] = (rho[(int)(get_global_id(0))] * (nuMix[(int)(get_global_id(0))] + nut[(int)(get_global_id(0))]));
  }
}
