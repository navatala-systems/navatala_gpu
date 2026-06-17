// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_sparse_knn` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_sparse_knn_sparse_k_n_n_distance_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_knn_sparse_k_n_n_distance_f32(device const float* queryData [[buffer(0)]], device const int* queryIndices [[buffer(1)]], device const uint* queryStart [[buffer(2)]], device const uint* queryEnd [[buffer(3)]], device const float* idxData [[buffer(4)]], device const int* idxIndices [[buffer(5)]], device const int* idxIndptr [[buffer(6)]], device const uint* nIdxRows [[buffer(7)]], device float* distances [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nRows = nIdxRows[0];
  uint qStart = queryStart[0];
  uint qEnd = queryEnd[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    uint rowIdxPlus1 = (rowIdx + 1u);
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    uint idxStart = ((uint)(idxRowStart));
    uint idxEnd = ((uint)(idxRowEnd));
    float queryNormSqAccum = as_type<float>(0x00000000u);
    uint qIdxAccum = qStart;
    uint qLen = (qEnd - qStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      uint qPos = qIdxAccum;
      if ((qPos < qEnd)) {
        float qVal = queryData[qPos];
        float qValSq = (qVal * qVal);
        float currQNorm = queryNormSqAccum;
        float newQNorm = (currQNorm + qValSq);
        queryNormSqAccum = newQNorm;
      }
      uint currQIdx = qIdxAccum;
      uint nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    float queryNormSq = queryNormSqAccum;
    float idxNormSqAccum = as_type<float>(0x00000000u);
    uint iIdxAccum = idxStart;
    uint idxLen = (idxEnd - idxStart);
    for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
      uint iPos = iIdxAccum;
      if ((iPos < idxEnd)) {
        float iVal = idxData[iPos];
        float iValSq = (iVal * iVal);
        float currINorm = idxNormSqAccum;
        float newINorm = (currINorm + iValSq);
        idxNormSqAccum = newINorm;
      }
      uint currIIdx = iIdxAccum;
      uint nextIIdx = (currIIdx + 1u);
      iIdxAccum = nextIIdx;
    }
    float idxNormSq = idxNormSqAccum;
    float dotAccum = as_type<float>(0x00000000u);
    uint qIdx2Accum = qStart;
    for (int dotQLoop = 0; dotQLoop < (int)(qLen); ++dotQLoop) {
      uint qPos2 = qIdx2Accum;
      if ((qPos2 < qEnd)) {
        float qVal2 = queryData[qPos2];
        int qCol = queryIndices[qPos2];
        uint iIdx2Accum = idxStart;
        for (int dotILoop = 0; dotILoop < (int)(idxLen); ++dotILoop) {
          uint iPos2 = iIdx2Accum;
          if ((iPos2 < idxEnd)) {
            int iCol = idxIndices[iPos2];
            bool colMatch = (qCol == iCol);
            if (colMatch) {
              float iVal2 = idxData[iPos2];
              float prod = (qVal2 * iVal2);
              float currDot = dotAccum;
              float newDot = (currDot + prod);
              dotAccum = newDot;
            }
          }
          uint currI2Idx = iIdx2Accum;
          uint nextI2Idx = (currI2Idx + 1u);
          iIdx2Accum = nextI2Idx;
        }
      }
      uint currQ2Idx = qIdx2Accum;
      uint nextQ2Idx = (currQ2Idx + 1u);
      qIdx2Accum = nextQ2Idx;
    }
    float dotProd = dotAccum;
    float twoDot = (as_type<float>(0x40000000u) * dotProd);
    float sumNorms = (queryNormSq + idxNormSq);
    float distSq = (sumNorms - twoDot);
    distances[rowIdx] = distSq;
  }
}

)kernel";
const char* k_metal_navatala_sparse_knn_sparse_inner_product_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_knn_sparse_inner_product_f32(device const float* queryData [[buffer(0)]], device const int* queryIndices [[buffer(1)]], device const uint* queryStart [[buffer(2)]], device const uint* queryEnd [[buffer(3)]], device const float* idxData [[buffer(4)]], device const int* idxIndices [[buffer(5)]], device const int* idxIndptr [[buffer(6)]], device const uint* nIdxRows [[buffer(7)]], device float* dotProducts [[buffer(8)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint nRows = nIdxRows[0];
  uint qStart = queryStart[0];
  uint qEnd = queryEnd[0];
  uint rowIdx = gid;
  bool inBounds = (rowIdx < nRows);
  if (inBounds) {
    int idxRowStart = idxIndptr[rowIdx];
    uint rowIdxPlus1 = (rowIdx + 1u);
    int idxRowEnd = idxIndptr[rowIdxPlus1];
    uint idxStart = ((uint)(idxRowStart));
    uint idxEnd = ((uint)(idxRowEnd));
    float dotAccum = as_type<float>(0x00000000u);
    uint qIdxAccum = qStart;
    uint qLen = (qEnd - qStart);
    uint idxLen = (idxEnd - idxStart);
    for (int qLoop = 0; qLoop < (int)(qLen); ++qLoop) {
      uint qPos = qIdxAccum;
      if ((qPos < qEnd)) {
        float qVal = queryData[qPos];
        int qCol = queryIndices[qPos];
        uint iIdxAccum = idxStart;
        for (int iLoop = 0; iLoop < (int)(idxLen); ++iLoop) {
          uint iPos = iIdxAccum;
          if ((iPos < idxEnd)) {
            int iCol = idxIndices[iPos];
            bool colMatch = (qCol == iCol);
            if (colMatch) {
              float iVal = idxData[iPos];
              float prod = (qVal * iVal);
              float currDot = dotAccum;
              float newDot = (currDot + prod);
              dotAccum = newDot;
            }
          }
          uint currIIdx = iIdxAccum;
          uint nextIIdx = (currIIdx + 1u);
          iIdxAccum = nextIIdx;
        }
      }
      uint currQIdx = qIdxAccum;
      uint nextQIdx = (currQIdx + 1u);
      qIdxAccum = nextQIdx;
    }
    float dotProd = dotAccum;
    dotProducts[rowIdx] = dotProd;
  }
}

)kernel";
const char* k_metal_navatala_sparse_knn_sparse_top_k_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_knn_sparse_top_k_f32(device const float* distances [[buffer(0)]], device const uint* nIdxRows [[buffer(1)]], device const uint* k [[buffer(2)]], device float* outDistances [[buffer(3)]], device uint* outIndices [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    uint nRows = nIdxRows[0];
    uint kVal = k[0];
    for (int initIdx = 0; initIdx < (int)(kVal); ++initIdx) {
      uint initIdxU32 = ((uint)(initIdx));
      outDistances[initIdxU32] = as_type<float>(0x7f7fc99eu);
      outIndices[initIdxU32] = 4294967295u;
    }
    for (int i = 0; i < (int)(nRows); ++i) {
      uint iU32 = ((uint)(i));
      float dist = distances[iU32];
      uint lastIdx = (kVal - 1u);
      float kthDist = outDistances[lastIdx];
      bool shouldInsert = (dist < kthDist);
      if (shouldInsert) {
        uint insertPosAccum = lastIdx;
        for (int scanIdx = 0; scanIdx < (int)(kVal); ++scanIdx) {
          uint scanIdxU32 = ((uint)(scanIdx));
          uint currentInsertPos = insertPosAccum;
          uint checkPos = (lastIdx - scanIdxU32);
          float checkDist = outDistances[checkPos];
          bool isSmaller = (dist < checkDist);
          uint newInsertPos = ((isSmaller) ? (checkPos) : (currentInsertPos));
          insertPosAccum = newInsertPos;
        }
        uint finalInsertPos = insertPosAccum;
        for (int shiftIdx = 0; shiftIdx < (int)(kVal); ++shiftIdx) {
          uint shiftIdxU32 = ((uint)(shiftIdx));
          uint shiftPos = (lastIdx - shiftIdxU32);
          bool shouldShift = (shiftPos > finalInsertPos);
          if (shouldShift) {
            uint srcPos = (shiftPos - 1u);
            float srcDist = outDistances[srcPos];
            uint srcIdx = outIndices[srcPos];
            outDistances[shiftPos] = srcDist;
            outIndices[shiftPos] = srcIdx;
          }
        }
        outDistances[finalInsertPos] = dist;
        outIndices[finalInsertPos] = iU32;
      }
    }
  }
}

)kernel";
const char* k_metal_navatala_sparse_knn_csr_row_distance_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_sparse_knn_csr_row_distance_f32(device const float* sparseData [[buffer(0)]], device const int* sparseIndices [[buffer(1)]], device const uint* sparseStart [[buffer(2)]], device const uint* sparseEnd [[buffer(3)]], device const float* denseQuery [[buffer(4)]], device const uint* nCols [[buffer(5)]], device float* distance [[buffer(6)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  uint sStart = sparseStart[0];
  uint sEnd = sparseEnd[0];
  uint nColsVal = nCols[0];
  threadgroup float sdata[256];
  float sparseNormAccum = as_type<float>(0x00000000u);
  float dotProdAccum = as_type<float>(0x00000000u);
  uint sIdxAccum = sStart;
  uint gridSize = ((uint)(int(__grid_size.x)));
  uint sparseLen = (sEnd - sStart);
  for (int sLoop = 0; sLoop < (int)(sparseLen); ++sLoop) {
    uint sPos = sIdxAccum;
    if ((sPos < sEnd)) {
      float sVal = sparseData[sPos];
      int sCol = sparseIndices[sPos];
      uint sColU32 = ((uint)(sCol));
      float sValSq = (sVal * sVal);
      float currSNorm = sparseNormAccum;
      float newSNorm = (currSNorm + sValSq);
      sparseNormAccum = newSNorm;
      float dVal = denseQuery[sColU32];
      float prod = (sVal * dVal);
      float currDot = dotProdAccum;
      float newDot = (currDot + prod);
      dotProdAccum = newDot;
    }
    uint currSIdx = sIdxAccum;
    uint nextSIdx = (currSIdx + 1u);
    sIdxAccum = nextSIdx;
  }
  float denseNormAccum = as_type<float>(0x00000000u);
  uint dIdxAccum = 0u;
  for (int dLoop = 0; dLoop < (int)(nColsVal); ++dLoop) {
    uint dPos = dIdxAccum;
    if ((dPos < nColsVal)) {
      float dValLoop = denseQuery[dPos];
      float dValSq = (dValLoop * dValLoop);
      float currDNorm = denseNormAccum;
      float newDNorm = (currDNorm + dValSq);
      denseNormAccum = newDNorm;
    }
    uint currDIdx = dIdxAccum;
    uint nextDIdx = (currDIdx + 1u);
    dIdxAccum = nextDIdx;
  }
  float sparseNormSq = sparseNormAccum;
  float denseNormSq = denseNormAccum;
  float dotProd = dotProdAccum;
  float twoDot = (as_type<float>(0x40000000u) * dotProd);
  float sumNorms = (sparseNormSq + denseNormSq);
  float distSq = (sumNorms - twoDot);
  bool isThread0 = (gid == 0u);
  if (isThread0) {
    distance[0u] = distSq;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_knn_sparse_k_n_n_distance_f32[] = {
  { "queryData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "queryEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "idxData", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndptr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distances", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_knn_sparse_k_n_n_distance_f32 = {
  1,
  "navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "metal",
  "navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "kernel:metal:navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "abi-r1:metal:navatala_sparse_knn_sparse_k_n_n_distance_f32",
  "abi-r1:metal:navatala_sparse_knn_sparse_k_n_n_distance_f32",
  9,
  kAbiArgs_metal_navatala_sparse_knn_sparse_k_n_n_distance_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_knn_sparse_inner_product_f32[] = {
  { "queryData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "queryStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "queryEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "idxData", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndices", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "idxIndptr", 6, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 7, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "dotProducts", 8, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_knn_sparse_inner_product_f32 = {
  1,
  "navatala_sparse_knn_sparse_inner_product_f32",
  "metal",
  "navatala_sparse_knn_sparse_inner_product_f32",
  "kernel:metal:navatala_sparse_knn_sparse_inner_product_f32",
  "abi-r1:metal:navatala_sparse_knn_sparse_inner_product_f32",
  "abi-r1:metal:navatala_sparse_knn_sparse_inner_product_f32",
  9,
  kAbiArgs_metal_navatala_sparse_knn_sparse_inner_product_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_knn_sparse_top_k_f32[] = {
  { "distances", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nIdxRows", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "outDistances", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "outIndices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_knn_sparse_top_k_f32 = {
  1,
  "navatala_sparse_knn_sparse_top_k_f32",
  "metal",
  "navatala_sparse_knn_sparse_top_k_f32",
  "kernel:metal:navatala_sparse_knn_sparse_top_k_f32",
  "abi-r1:metal:navatala_sparse_knn_sparse_top_k_f32",
  "abi-r1:metal:navatala_sparse_knn_sparse_top_k_f32",
  5,
  kAbiArgs_metal_navatala_sparse_knn_sparse_top_k_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_sparse_knn_csr_row_distance_f32[] = {
  { "sparseData", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseIndices", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "sparseStart", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "sparseEnd", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "denseQuery", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nCols", 5, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "distance", 6, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_sparse_knn_csr_row_distance_f32 = {
  1,
  "navatala_sparse_knn_csr_row_distance_f32",
  "metal",
  "navatala_sparse_knn_csr_row_distance_f32",
  "kernel:metal:navatala_sparse_knn_csr_row_distance_f32",
  "abi-r1:metal:navatala_sparse_knn_csr_row_distance_f32",
  "abi-r1:metal:navatala_sparse_knn_csr_row_distance_f32",
  7,
  kAbiArgs_metal_navatala_sparse_knn_csr_row_distance_f32
};

bool tryGetKernelAbiManifest_metal_sparse_knn(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f32") {
    out = &kAbiManifest_metal_navatala_sparse_knn_sparse_k_n_n_distance_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_knn_sparse_inner_product_f32") {
    out = &kAbiManifest_metal_navatala_sparse_knn_sparse_inner_product_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_knn_sparse_top_k_f32") {
    out = &kAbiManifest_metal_navatala_sparse_knn_sparse_top_k_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_knn_csr_row_distance_f32") {
    out = &kAbiManifest_metal_navatala_sparse_knn_csr_row_distance_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_sparse_knn(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_sparse_knn_sparse_k_n_n_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_knn_sparse_k_n_n_distance_f32";
    std::string_view sv(k_metal_navatala_sparse_knn_sparse_k_n_n_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_knn_sparse_inner_product_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_knn_sparse_inner_product_f32";
    std::string_view sv(k_metal_navatala_sparse_knn_sparse_inner_product_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_knn_sparse_top_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_knn_sparse_top_k_f32";
    std::string_view sv(k_metal_navatala_sparse_knn_sparse_top_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_sparse_knn_csr_row_distance_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_sparse_knn_csr_row_distance_f32";
    std::string_view sv(k_metal_navatala_sparse_knn_csr_row_distance_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

