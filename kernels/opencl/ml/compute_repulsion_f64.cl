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
__kernel void navatala_ml_compute_repulsion_f64(__global const double* embedding, __global const int* negative_samples, __global const uint* n_samples, __global const uint* n_neg, __global const uint* n_components, __global const double* a_param, __global const double* b_param, __global const double* gamma, __global double* gradients) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint nSamples = n_samples[0];
  uint nNeg = n_neg[0];
  uint nComp = n_components[0];
  double a = a_param[0];
  double b = b_param[0];
  double gammaVal = gamma[0];
  bool inBounds = (gid < nSamples);
  if (inBounds) {
    uint sampleBase = (gid * nComp);
    uint negBase = (gid * nNeg);
    for (int n = 0; n < (int)(nNeg); ++n) {
      uint nU32 = ((uint)(n));
      uint negSampleIdx = (negBase + nU32);
      int negIdx = negative_samples[negSampleIdx];
      bool validNeg = (negIdx >= 0);
      if (validNeg) {
        uint negIdxU32 = ((uint)(negIdx));
        uint negEmbBase = (negIdxU32 * nComp);
        double distSqNeg = as_double(0x3f1a36e2eb1c432dul);
        for (int d = 0; d < (int)(nComp); ++d) {
          uint dU32 = ((uint)(d));
          uint iAddr = (sampleBase + dU32);
          uint jAddr = (negEmbBase + dU32);
          double iVal = embedding[iAddr];
          double jVal = embedding[jAddr];
          double diff = (iVal - jVal);
          double diffSq = (diff * diff);
          double currDSq = distSqNeg;
          double newDSq = (currDSq + diffSq);
          distSqNeg = newDSq;
        }
        double dSq = distSqNeg;
        double logDSqNeg = log(dSq);
        double dPow2b = exp((b * logDSqNeg));
        double denom = (as_double(0x3ff0000000000000ul) + (a * dPow2b));
        double gradScale = (((as_double(0x4000000000000000ul) * gammaVal) * b) / (dSq * denom));
        for (int d = 0; d < (int)(nComp); ++d) {
          uint dU32 = ((uint)(d));
          uint iAddr = (sampleBase + dU32);
          uint jAddr = (negEmbBase + dU32);
          double iVal = embedding[iAddr];
          double jVal = embedding[jAddr];
          double diff = (iVal - jVal);
          double grad = (gradScale * diff);
          double gradClamped = (((grad > as_double(0x4010000000000000ul))) ? (as_double(0x4010000000000000ul)) : ((((grad < as_double(0xc010000000000000ul))) ? (as_double(0xc010000000000000ul)) : (grad))));
          double currGrad = gradients[iAddr];
          double newGrad = (currGrad + gradClamped);
          gradients[iAddr] = newGrad;
        }
      }
    }
  }
}
