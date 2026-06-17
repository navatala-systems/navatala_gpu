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

__kernel void navatala_samples_axpy_fallback(__global const float* x, __global const float* y, __global const float* alpha, __global float* _out) {
  int gid0 = (int)get_global_id(0);
  _out[(int)(get_global_id(0))] = (y[(int)(get_global_id(0))] + (x[(int)(get_global_id(0))] * alpha[0]));
}
