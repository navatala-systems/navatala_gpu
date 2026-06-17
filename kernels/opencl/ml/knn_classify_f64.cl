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

__kernel void navatala_ml_knn_classify_f64(__global const uint* neighborIndices, __global const uint* labels, __global const uint* q, __global const uint* k, __global const uint* numClasses, __global uint* predictions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint qVal = q[0];
  uint kVal = k[0];
  uint nClasses = numClasses[0];
  uint queryIdx = gid;
  bool inBounds = (queryIdx < qVal);
  if (inBounds) {
    uint baseIdx = (queryIdx * kVal);
    uint bestClass = (uint)(0u);
    uint bestVotes = (uint)(0u);
    for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
      uint classIdxU32 = ((uint)(classIdx));
      uint classVotes = (uint)(0u);
      for (int j = 0; j < (int)(kVal); ++j) {
        uint jU32 = ((uint)(j));
        uint neighborIdx = (baseIdx + jU32);
        uint neighborDataIdx = neighborIndices[neighborIdx];
        uint neighborLabel = labels[neighborDataIdx];
        bool isMatch = (neighborLabel == classIdxU32);
        uint increment = ((isMatch) ? ((uint)(1u)) : ((uint)(0u)));
        uint currentVotes = classVotes;
        uint newClassVotes = (currentVotes + increment);
        classVotes = newClassVotes;
      }
      uint finalClassVotes = classVotes;
      uint currentBestVotes = bestVotes;
      uint currentBestClass = bestClass;
      bool isBetter = (finalClassVotes > currentBestVotes);
      uint newBestClass = ((isBetter) ? (classIdxU32) : (currentBestClass));
      uint newBestVotes = ((isBetter) ? (finalClassVotes) : (currentBestVotes));
      bestClass = newBestClass;
      bestVotes = newBestVotes;
    }
    uint finalBestClass = bestClass;
    predictions[queryIdx] = finalBestClass;
  }
}
