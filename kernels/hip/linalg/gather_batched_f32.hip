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

#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_linalg_gather_batched_f32(const float* A, const unsigned int* indices, const unsigned int* batchSize, const unsigned int* m, const unsigned int* n, const unsigned int* kCount, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int tid = gid;
  unsigned int batchVal = batchSize[0u];
  unsigned int mVal = m[0u];
  unsigned int nVal = n[0u];
  unsigned int kVal = kCount[0u];
  unsigned int outBatchStride = (kVal * nVal);
  unsigned int totalElems = (batchVal * outBatchStride);
  if ((tid < totalElems)) {
    unsigned int batchIdx = (tid / outBatchStride);
    unsigned int withinBatch = (tid % outBatchStride);
    unsigned int rowIdx = (withinBatch / nVal);
    unsigned int colIdx = (withinBatch % nVal);
    unsigned int indicesIdx = ((batchIdx * kVal) + rowIdx);
    unsigned int srcRow = indices[indicesIdx];
    unsigned int srcBatchStride = (mVal * nVal);
    unsigned int srcIdx = (((batchIdx * srcBatchStride) + (srcRow * nVal)) + colIdx);
    float val = A[srcIdx];
    _out[tid] = val;
  }
}
