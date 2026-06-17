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

#include <cuda_runtime.h>
extern "C" __global__ void navatala_dataframe_select_top_k_f32(const float* values, const unsigned int* count, const unsigned int* k, float* outValues, unsigned int* outIndices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int countVal = count[0];
  unsigned int kVal = k[0];
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      unsigned int initIdxU32 = ((unsigned int)(initIdx));
      outValues[initIdxU32] = __uint_as_float(0x7f7fc99eu);
      outIndices[initIdxU32] = 4294967295u;
    }
    for (int i = 0; i < (int)(countVal); ++i) {
      unsigned int iU32 = ((unsigned int)(i));
      float val = values[iU32];
      unsigned int lastIdx = (kVal - 1u);
      float kthVal = outValues[lastIdx];
      bool shouldInsert = (val < kthVal);
      if (shouldInsert) {
        unsigned int insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          unsigned int scanIdxU32 = ((unsigned int)(scanIdx));
          unsigned int currentInsertPos = insertPosAccum;
          unsigned int checkPos = (lastIdx - scanIdxU32);
          float checkVal = outValues[checkPos];
          bool isSmaller = (val < checkVal);
          unsigned int newInsertPos = ((isSmaller) ? (checkPos) : (currentInsertPos));
          insertPosAccum = newInsertPos;
        }
        unsigned int finalInsertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)(kVal); ++shiftIdx) {
          unsigned int shiftIdxU32 = ((unsigned int)(shiftIdx));
          unsigned int shiftPos = (lastIdx - shiftIdxU32);
          bool shouldShift = (shiftPos > finalInsertPos);
          if (shouldShift) {
            unsigned int srcPos = (shiftPos - 1u);
            float srcVal = outValues[srcPos];
            unsigned int srcIdx = outIndices[srcPos];
            outValues[shiftPos] = srcVal;
            outIndices[shiftPos] = srcIdx;
          }
        }
        outValues[finalInsertPos] = val;
        outIndices[finalInsertPos] = iU32;
      }
    }
  }
}
