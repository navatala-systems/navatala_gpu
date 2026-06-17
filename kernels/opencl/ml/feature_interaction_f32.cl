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

__kernel void navatala_ml_feature_interaction_f32(__global const float* jointContrib, __global const float* shapValues, __global const uint* nFeatures, __global float* interactions) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nFeaturesVal = nFeatures[0];
  uint totalElems = (nFeaturesVal * nFeaturesVal);
  bool inBounds = (gid < totalElems);
  if (inBounds) {
    uint i = (gid / nFeaturesVal);
    uint j = (gid % nFeaturesVal);
    float phiIJ = jointContrib[gid];
    float phiI = shapValues[i];
    float phiJ = shapValues[j];
    float sumIndividual = (phiI + phiJ);
    float interaction = (phiIJ - sumIndividual);
    interactions[gid] = interaction;
  }
}
