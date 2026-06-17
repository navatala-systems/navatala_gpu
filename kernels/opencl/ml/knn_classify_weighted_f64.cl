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

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_ml_knn_classify_weighted_f64(__global const uint* neighborIndices, __global const double* neighborDistances, __global const uint* labels, __global const uint* q, __global const uint* k, __global const uint* numClasses, __global uint* predictions) {
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
    double bestWeight = as_double(0x0000000000000000ul);
    double epsilon = as_double(0x3cd203af9ee75616ul);
    for (int classIdx = 0; classIdx < (int)(nClasses); ++classIdx) {
      uint classIdxU32 = ((uint)(classIdx));
      double classWeight = as_double(0x0000000000000000ul);
      for (int j = 0; j < (int)(kVal); ++j) {
        uint jU32 = ((uint)(j));
        uint neighborIdx = (baseIdx + jU32);
        uint neighborDataIdx = neighborIndices[neighborIdx];
        uint neighborLabel = labels[neighborDataIdx];
        double dist = neighborDistances[neighborIdx];
        double weight = (as_double(0x3ff0000000000000ul) / (dist + epsilon));
        bool isMatch = (neighborLabel == classIdxU32);
        double addWeight = ((isMatch) ? (weight) : (as_double(0x0000000000000000ul)));
        double currentWeight = classWeight;
        double newClassWeight = (currentWeight + addWeight);
        classWeight = newClassWeight;
      }
      double finalClassWeight = classWeight;
      double currentBestWeight = bestWeight;
      uint currentBestClass = bestClass;
      bool isBetter = (finalClassWeight > currentBestWeight);
      uint newBestClass = ((isBetter) ? (classIdxU32) : (currentBestClass));
      double newBestWeight = ((isBetter) ? (finalClassWeight) : (currentBestWeight));
      bestClass = newBestClass;
      bestWeight = newBestWeight;
    }
    uint finalBestClass = bestClass;
    predictions[queryIdx] = finalBestClass;
  }
}
