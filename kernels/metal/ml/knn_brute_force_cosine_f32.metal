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

#include <metal_stdlib>
using namespace metal;

kernel void navatala_ml_knn_brute_force_cosine_f32(device const float* queries [[buffer(0)]], device const float* data [[buffer(1)]], device const uint* q [[buffer(2)]], device const uint* n [[buffer(3)]], device const uint* d [[buffer(4)]], device const uint* k [[buffer(5)]], device uint* indices [[buffer(6)]], device float* distances [[buffer(7)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint qVal = q[0];
  uint nVal = n[0];
  uint dVal = d[0];
  uint kVal = k[0];
  bool inBounds = (gid < qVal);
  if (inBounds) {
    uint queryBase = (gid * dVal);
    uint outBase = (gid * kVal);
    float queryNormSqAccum = as_type<float>(0x00000000u);
    for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
      uint dimIdxU32 = ((uint)(dimIdx));
      uint qIdx = (queryBase + dimIdxU32);
      float qv = queries[qIdx];
      float qvSq = (qv * qv);
      float currQueryNormSq = queryNormSqAccum;
      float newQueryNormSq = (currQueryNormSq + qvSq);
      queryNormSqAccum = newQueryNormSq;
    }
    float queryNormSq = queryNormSqAccum;
    float queryNorm = sqrt(queryNormSq);
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      uint outIdx = (outBase + initIdxU32);
      distances[outIdx] = as_type<float>(0x7f7fc99eu);
      indices[outIdx] = 4294967295u;
    }
    for (int dataIdx = 0; dataIdx < (int)(nVal); ++dataIdx) {
      uint dataIdxU32 = ((uint)(dataIdx));
      uint dataBase = (dataIdxU32 * dVal);
      float dotProdAccum = as_type<float>(0x00000000u);
      float dataNormSqAccum = as_type<float>(0x00000000u);
      for (int dimIdx = 0; dimIdx < (int)(dVal); ++dimIdx) {
        uint dimIdxU32 = ((uint)(dimIdx));
        uint qIdx = (queryBase + dimIdxU32);
        uint dIdx = (dataBase + dimIdxU32);
        float qv = queries[qIdx];
        float dv = data[dIdx];
        float prod = (qv * dv);
        float currDotProd = dotProdAccum;
        float newDotProd = (currDotProd + prod);
        dotProdAccum = newDotProd;
        float dvSq = (dv * dv);
        float currDataNormSq = dataNormSqAccum;
        float newDataNormSq = (currDataNormSq + dvSq);
        dataNormSqAccum = newDataNormSq;
      }
      float dotProd = dotProdAccum;
      float dataNormSq = dataNormSqAccum;
      float dataNorm = sqrt(dataNormSq);
      float normProduct = (queryNorm * dataNorm);
      bool normIsZero = (normProduct == as_type<float>(0x00000000u));
      float cosineSim = (dotProd / normProduct);
      float cosineDist = (as_type<float>(0x3f800000u) - cosineSim);
      float dist = ((normIsZero) ? (as_type<float>(0x3f800000u)) : (cosineDist));
      uint worstIdx = (outBase + (kVal - 1u));
      float worstDist = distances[worstIdx];
      bool shouldInsert = (dist < worstDist);
      if (shouldInsert) {
        uint initInsertPos = (kVal - 1u);
        uint insertPosAccum = initInsertPos;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint checkPos = ((kVal - 1u) - scanIdxU32);
          uint checkIdx = (outBase + checkPos);
          float checkDist = distances[checkIdx];
          bool shouldShift = (dist < checkDist);
          uint currInsertPos = insertPosAccum;
          uint newInsertPos = ((shouldShift) ? (checkPos) : (currInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint insertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)((kVal - 1u)); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = ((kVal - 2u) - shiftIdxU32);
          bool shouldDoShift = (shiftPos >= insertPos);
          if (shouldDoShift) {
            uint srcIdx = (outBase + shiftPos);
            uint dstIdx = (outBase + (shiftPos + 1u));
            float shiftDist = distances[srcIdx];
            uint shiftIndex = indices[srcIdx];
            distances[dstIdx] = shiftDist;
            indices[dstIdx] = shiftIndex;
          }
        }
        uint insertIdx = (outBase + insertPos);
        distances[insertIdx] = dist;
        indices[insertIdx] = dataIdxU32;
      }
    }
  }
}
