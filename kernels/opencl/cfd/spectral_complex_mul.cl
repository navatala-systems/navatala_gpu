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

__kernel void navatala_cfd_spectral_complex_mul(__global const float* x, __global const float* w, __global const int* n, __global float* y) {
  int gid0 = (int)get_global_id(0);
  int gid = (int)(get_global_id(0));
  int count = n[0];
  if ((gid < count)) {
    float xr = x[((gid)*2 + 0)];
    float xi = x[((gid)*2 + 1)];
    float wr = w[((gid)*2 + 0)];
    float wi = w[((gid)*2 + 1)];
    float yr = ((xr * wr) - (xi * wi));
    float yi = ((xr * wi) + (xi * wr));
    y[((gid)*2 + 0)] = yr;
    y[((gid)*2 + 1)] = yi;
  }
}
