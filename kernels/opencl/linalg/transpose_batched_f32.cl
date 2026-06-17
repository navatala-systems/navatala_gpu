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

__kernel void navatala_linalg_transpose_batched_f32(__global const float* A, __global const uint* m, __global const uint* n, __global const uint* batchSize, __global float* B) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint rows = m[0];
  uint cols = n[0];
  uint numBatches = batchSize[0];
  uint matSize = (rows * cols);
  uint totalElems = (matSize * numBatches);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint batchIdx = (gid / matSize);
    uint localIdx = (gid % matSize);
    uint i = (localIdx / cols);
    uint j = (localIdx % cols);
    uint srcIdx = ((batchIdx * matSize) + ((i * cols) + j));
    uint dstIdx = ((batchIdx * matSize) + ((j * rows) + i));
    float val = A[srcIdx];
    B[dstIdx] = val;
  }
}
