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
extern "C" __global__ void Solver_CFD_StandaloneCfd03MarineGpuKernels_hullWindowStatisticsF64(const double* times, const double* values, const unsigned int* count, double* scratchA, double* scratchB, double* firstTimeOut, double* lastTimeOut, double* meanOut, double* sampleStandardDeviationOut, double* minimumOut, double* maximumOut, double* driftSlopeOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int sampleCount = count[0];
  if (gid == 0u) {
    double firstTime = times[0u];
    unsigned int lastIndex = (sampleCount - 1u);
    double lastTime = times[lastIndex];
    double minimum = values[0u];
    double maximum = values[0u];
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      double sampleValue = values[sampleIndex];
      scratchA[sampleIndex] = sampleValue;
      if (sampleValue < minimum) {
        minimum = sampleValue;
      }
      if (maximum < sampleValue) {
        maximum = sampleValue;
      }
    }
    unsigned int valueActive = sampleCount;
    unsigned int valueCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < valueActive) {
        unsigned int pairs = (valueActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((valueCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((valueCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (valueCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (valueActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (valueActive - 1u);
          double carryValue = (((valueCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (valueCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        valueActive = (pairs + odd);
        valueCurrentA = (((valueCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double valueSum = (((valueCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    double sampleCountF64 = ((double)(sampleCount));
    double mean = (valueSum / sampleCountF64);
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      scratchA[sampleIndex] = times[sampleIndex];
    }
    unsigned int timeActive = sampleCount;
    unsigned int timeCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < timeActive) {
        unsigned int pairs = (timeActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((timeCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((timeCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (timeCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (timeActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (timeActive - 1u);
          double carryValue = (((timeCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (timeCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        timeActive = (pairs + odd);
        timeCurrentA = (((timeCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double timeSum = (((timeCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    double meanTime = (timeSum / sampleCountF64);
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      double deviation = (values[sampleIndex] - mean);
      scratchA[sampleIndex] = (deviation * deviation);
    }
    unsigned int varianceActive = sampleCount;
    unsigned int varianceCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < varianceActive) {
        unsigned int pairs = (varianceActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((varianceCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((varianceCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (varianceCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (varianceActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (varianceActive - 1u);
          double carryValue = (((varianceCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (varianceCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        varianceActive = (pairs + odd);
        varianceCurrentA = (((varianceCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double varianceSum = (((varianceCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    double sampleDenominator = ((double)((sampleCount - 1u)));
    double sampleStandardDeviation = sqrt((varianceSum / sampleDenominator));
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      double timeDeviation = (times[sampleIndex] - meanTime);
      double valueDeviation = (values[sampleIndex] - mean);
      scratchA[sampleIndex] = (timeDeviation * valueDeviation);
    }
    unsigned int covarianceActive = sampleCount;
    unsigned int covarianceCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < covarianceActive) {
        unsigned int pairs = (covarianceActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((covarianceCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((covarianceCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (covarianceCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (covarianceActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (covarianceActive - 1u);
          double carryValue = (((covarianceCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (covarianceCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        covarianceActive = (pairs + odd);
        covarianceCurrentA = (((covarianceCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double covarianceSum = (((covarianceCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    for (int _sample = 0; _sample < (int)(sampleCount); ++_sample) {
      unsigned int sampleIndex = ((unsigned int)(_sample));
      double timeDeviation = (times[sampleIndex] - meanTime);
      scratchA[sampleIndex] = (timeDeviation * timeDeviation);
    }
    unsigned int timeVarianceActive = sampleCount;
    unsigned int timeVarianceCurrentA = 1u;
    for (int reductionLevel = 0; reductionLevel < (int)(32u); ++reductionLevel) {
      if (1u < timeVarianceActive) {
        unsigned int pairs = (timeVarianceActive / 2u);
        for (int pair = 0; pair < (int)(pairs); ++pair) {
          unsigned int pairIndex = ((unsigned int)(pair));
          unsigned int leftIndex = (pairIndex * 2u);
          unsigned int rightIndex = (leftIndex + 1u);
          double leftValue = (((timeVarianceCurrentA == 1u)) ? (scratchA[leftIndex]) : (scratchB[leftIndex]));
          double rightValue = (((timeVarianceCurrentA == 1u)) ? (scratchA[rightIndex]) : (scratchB[rightIndex]));
          double pairValue = (leftValue + rightValue);
          if (timeVarianceCurrentA == 1u) {
            scratchB[pairIndex] = pairValue;
          } else {
            scratchA[pairIndex] = pairValue;
          }
        }
        unsigned int odd = (timeVarianceActive % 2u);
        if (odd == 1u) {
          unsigned int carryIndex = (timeVarianceActive - 1u);
          double carryValue = (((timeVarianceCurrentA == 1u)) ? (scratchA[carryIndex]) : (scratchB[carryIndex]));
          if (timeVarianceCurrentA == 1u) {
            scratchB[pairs] = carryValue;
          } else {
            scratchA[pairs] = carryValue;
          }
        }
        timeVarianceActive = (pairs + odd);
        timeVarianceCurrentA = (((timeVarianceCurrentA == 1u)) ? (0u) : (1u));
      }
    }
    double timeVarianceSum = (((timeVarianceCurrentA == 1u)) ? (scratchA[0u]) : (scratchB[0u]));
    double driftSlope = (covarianceSum / timeVarianceSum);
    firstTimeOut[0u] = firstTime;
    lastTimeOut[0u] = lastTime;
    meanOut[0u] = mean;
    sampleStandardDeviationOut[0u] = sampleStandardDeviation;
    minimumOut[0u] = minimum;
    maximumOut[0u] = maximum;
    driftSlopeOut[0u] = driftSlope;
  }
}
