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
extern "C" __global__ void navatala_ml_propagate_core_labels_f32(const float* points, const int* labelsIn, const unsigned int* isCore, const float* epsilon, const unsigned int* nPoints, const unsigned int* nDims, int* labelsOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = nPoints[0];
  unsigned int d = nDims[0];
  float eps = epsilon[0];
  float epsSq = (eps * eps);
  bool inBounds = (gid < n);
  if (inBounds) {
    int myLabel = labelsIn[gid];
    unsigned int myIsCore = isCore[gid];
    int newLabelAccum = myLabel;
    bool notCore = (myIsCore == 0u);
    bool isUnlabeled = (myLabel == -1);
    bool shouldProcess = (notCore && isUnlabeled);
    if (shouldProcess) {
      unsigned int iBase = (gid * d);
      float bestDistAccum = __uint_as_float(0x7f7fffffu);
      for (int j = 0; j < (int)(n); ++j) {
        unsigned int jU32 = ((unsigned int)(j));
        unsigned int jIsCore = isCore[jU32];
        int jLabel = labelsIn[jU32];
        bool jIsCoreVal = (jIsCore == 1u);
        bool jHasLabel = (jLabel >= 0);
        bool jIsLabeledCore = (jIsCoreVal && jHasLabel);
        if (jIsLabeledCore) {
          unsigned int jBase = (jU32 * d);
          float distSqAccum = __uint_as_float(0x00000000u);
          for (int k = 0; k < (int)(d); ++k) {
            unsigned int kU32 = ((unsigned int)(k));
            unsigned int iIdx = (iBase + kU32);
            unsigned int jIdx = (jBase + kU32);
            float iVal = points[iIdx];
            float jVal = points[jIdx];
            float diff = (iVal - jVal);
            float diffSq = (diff * diff);
            float currentDistSq = distSqAccum;
            float newDistSq = (currentDistSq + diffSq);
            distSqAccum = newDistSq;
          }
          float finalDistSq = distSqAccum;
          float currentBestDist = bestDistAccum;
          bool withinEps = (finalDistSq < epsSq);
          bool isBetter = (finalDistSq < currentBestDist);
          bool shouldUpdate = (withinEps && isBetter);
          if (shouldUpdate) {
            bestDistAccum = finalDistSq;
            newLabelAccum = jLabel;
          }
        }
      }
    }
    int finalNewLabel = newLabelAccum;
    labelsOut[gid] = finalNewLabel;
  }
}
