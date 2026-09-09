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
__kernel void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64(__global const double* times, __global const double* values, __global const uint* count, __global double* scratchA, __global double* scratchB, __global double* firstTimeOut, __global double* lastTimeOut, __global double* meanOut, __global double* sampleStandardDeviationOut, __global double* minimumOut, __global double* maximumOut, __global double* driftSlopeOut) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint sampleCount = count[0];
  if (gid == (uint)(0u)) {
    double firstTime = times[(uint)(0u)];
    uint lastIndex = (sampleCount - (uint)(1u));
    double lastTime = times[lastIndex];
    double minimum = values[(uint)(0u)];
    double maximum = values[(uint)(0u)];
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      double sampleValue = values[sampleIndex];
      scratchA[sampleIndex] = sampleValue;
      if (sampleValue < minimum) {
        minimum = sampleValue;
      }
      if (maximum < sampleValue) {
        maximum = sampleValue;
      }
    }
    uint valueActive = sampleCount;
    uint valueCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < valueActive) {
        uint pairs = (valueActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((valueCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((valueCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (valueCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (valueActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (valueActive - (uint)(1u));
          double carryValue = (((valueCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (valueCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        valueActive = (pairs + odd);
        valueCurrentA = (((valueCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double valueSum = (((valueCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    double sampleCountF64 = ((double)(sampleCount));
    double mean = (valueSum / sampleCountF64);
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      scratchA[sampleIndex] = times[sampleIndex];
    }
    uint timeActive = sampleCount;
    uint timeCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < timeActive) {
        uint pairs = (timeActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((timeCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((timeCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (timeCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (timeActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (timeActive - (uint)(1u));
          double carryValue = (((timeCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (timeCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        timeActive = (pairs + odd);
        timeCurrentA = (((timeCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double timeSum = (((timeCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    double meanTime = (timeSum / sampleCountF64);
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      double deviation = (values[sampleIndex] - mean);
      scratchA[sampleIndex] = (deviation * deviation);
    }
    uint varianceActive = sampleCount;
    uint varianceCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < varianceActive) {
        uint pairs = (varianceActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((varianceCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((varianceCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (varianceCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (varianceActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (varianceActive - (uint)(1u));
          double carryValue = (((varianceCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (varianceCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        varianceActive = (pairs + odd);
        varianceCurrentA = (((varianceCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double varianceSum = (((varianceCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    double sampleDenominator = ((double)((sampleCount - (uint)(1u))));
    double sampleStandardDeviation = sqrt((varianceSum / sampleDenominator));
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      double timeDeviation = (times[sampleIndex] - meanTime);
      double valueDeviation = (values[sampleIndex] - mean);
      scratchA[sampleIndex] = (timeDeviation * valueDeviation);
    }
    uint covarianceActive = sampleCount;
    uint covarianceCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < covarianceActive) {
        uint pairs = (covarianceActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((covarianceCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((covarianceCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (covarianceCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (covarianceActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (covarianceActive - (uint)(1u));
          double carryValue = (((covarianceCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (covarianceCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        covarianceActive = (pairs + odd);
        covarianceCurrentA = (((covarianceCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double covarianceSum = (((covarianceCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      uint sampleIndex = ((uint)(_sample));
      double timeDeviation = (times[sampleIndex] - meanTime);
      scratchA[sampleIndex] = (timeDeviation * timeDeviation);
    }
    uint timeVarianceActive = sampleCount;
    uint timeVarianceCurrentA = (uint)(1u);
    for (int reductionLevel = 0; reductionLevel < (int)((uint)(32u)); ++reductionLevel) {
      if ((uint)(1u) < timeVarianceActive) {
        uint pairs = (timeVarianceActive / (uint)(2u));
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          uint pairIndex = ((uint)(pair));
          uint leftIndex = (pairIndex * (uint)(2u));
          uint rightIndex = (leftIndex + (uint)(1u));
          double leftValue = (((timeVarianceCurrentA == (uint)(1u))) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((timeVarianceCurrentA == (uint)(1u))) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (timeVarianceCurrentA == (uint)(1u)) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        uint odd = (timeVarianceActive % (uint)(2u));
        if (odd == (uint)(1u)) {
          uint carryIndex = (timeVarianceActive - (uint)(1u));
          double carryValue = (((timeVarianceCurrentA == (uint)(1u))) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (timeVarianceCurrentA == (uint)(1u)) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        timeVarianceActive = (pairs + odd);
        timeVarianceCurrentA = (((timeVarianceCurrentA == (uint)(1u))) ? ((uint)(0u)) : ((uint)(1u)));
      }
    }
    double timeVarianceSum = (((timeVarianceCurrentA == (uint)(1u))) ? (scratchA[(uint)(0u)]) : (scratchB[(uint)(0u)]));
    double driftSlope = (covarianceSum / timeVarianceSum);
    firstTimeOut[(uint)(0u)] = firstTime;
    lastTimeOut[(uint)(0u)] = lastTime;
    meanOut[(uint)(0u)] = mean;
    sampleStandardDeviationOut[(uint)(0u)] = sampleStandardDeviation;
    minimumOut[(uint)(0u)] = minimum;
    maximumOut[(uint)(0u)] = maximum;
    driftSlopeOut[(uint)(0u)] = driftSlope;
  }
}
