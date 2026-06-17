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

__kernel void navatala_sparse_mg_convergence_relative_f32(__global const float* residualNorm, __global const float* initialNorm, __global const float* tolerance, __global uint* converged) {
  int gid0 = (int)get_global_id(0);
  float norm = residualNorm[0];
  float init = initialNorm[0];
  float tol = tolerance[0];
  float ratio = (norm / init);
  uint flag = (((ratio < tol)) ? ((uint)(1u)) : ((uint)(0u)));
  converged[0] = flag;
}
