// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_runtime` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_runtime_deterministic_reduce_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_runtime_deterministic_reduce_f32(device const float* _input [[buffer(0)]], device const uint* count [[buffer(1)]], device float* result [[buffer(2)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float val = _input[gid];
    sdata[lid] = val;
  } else {
    sdata[lid] = as_type<float>(0x00000000u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint f32RedStride = 128u;
  for (int f32RedStep = 0; f32RedStep < (int)(8); ++f32RedStep) {
    uint f32Stride = f32RedStride;
    if (lid < f32Stride) {
      float other = sdata[(lid + f32Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint f32StrideToHalve = f32RedStride;
    uint f32NextStride = (f32StrideToHalve >> 1u);
    f32RedStride = f32NextStride;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0u) {
    float totalSum = sdata[0];
    result[0] = totalSum;
  }
}

)kernel";
const char* k_metal_navatala_runtime_deterministic_histogram_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_runtime_deterministic_histogram_f32(device const float* values [[buffer(0)]], device const uint* numBins [[buffer(1)]], device const float* minVal [[buffer(2)]], device const float* maxVal [[buffer(3)]], device const uint* count [[buffer(4)]], device uint* histogram [[buffer(5)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint lid = ((uint)(int(__tid.x)));
  threadgroup uint binIndices[256];
  threadgroup uint localHist[256];
  uint countVal = count[0];
  uint numBinsVal = numBins[0];
  float minValF = minVal[0];
  float maxValF = maxVal[0];
  float range = (maxValF - minValF);
  float numBinsFloat = ((float)(numBinsVal));
  float binWidth = (range / numBinsFloat);
  if (lid < numBinsVal) {
    localHist[lid] = 0u;
  }
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float val = values[gid];
    float normalized = ((val - minValF) / binWidth);
    float binIdxFloat = normalized;
    uint binIdx = ((uint)(binIdxFloat));
    uint maxBinIdx = (numBinsVal - 1u);
    uint clampedBinIdx = (((binIdx < numBinsVal)) ? (binIdx) : (maxBinIdx));
    binIndices[lid] = clampedBinIdx;
  } else {
    binIndices[lid] = numBinsVal;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < numBinsVal) {
    uint myBin = lid;
    uint histF32RedStride = 128u;
    for (int histF32RedStep = 0; histF32RedStep < (int)(8); ++histF32RedStep) {
      uint histF32Stride = histF32RedStride;
      if (lid < histF32Stride) {
        uint partnerIdx = (lid + histF32Stride);
        uint storedBin = binIndices[partnerIdx];
        if (storedBin == myBin) {
          uint oldCount = localHist[lid];
          uint newCount = (oldCount + 1u);
          localHist[lid] = newCount;
        }
      }
      uint histF32StrideToHalve = histF32RedStride;
      uint histF32NextStride = (histF32StrideToHalve >> 1u);
      histF32RedStride = histF32NextStride;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0u) {
    for (int bin = 0; bin < (int)(numBinsVal); ++bin) {
      uint binU32 = ((uint)(bin));
      uint localCount = localHist[binU32];
      uint globalCount = histogram[binU32];
      uint newGlobalCount = (globalCount + localCount);
      histogram[binU32] = newGlobalCount;
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_runtime_deterministic_reduce_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_runtime_deterministic_reduce_f32 = {
  1,
  "navatala_runtime_deterministic_reduce_f32",
  "metal",
  "navatala_runtime_deterministic_reduce_f32",
  "kernel:metal:navatala_runtime_deterministic_reduce_f32",
  "abi-r1:metal:navatala_runtime_deterministic_reduce_f32",
  "abi-r1:metal:navatala_runtime_deterministic_reduce_f32",
  3,
  kAbiArgs_metal_navatala_runtime_deterministic_reduce_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_runtime_deterministic_histogram_f32[] = {
  { "values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBins", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "minVal", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "histogram", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_runtime_deterministic_histogram_f32 = {
  1,
  "navatala_runtime_deterministic_histogram_f32",
  "metal",
  "navatala_runtime_deterministic_histogram_f32",
  "kernel:metal:navatala_runtime_deterministic_histogram_f32",
  "abi-r1:metal:navatala_runtime_deterministic_histogram_f32",
  "abi-r1:metal:navatala_runtime_deterministic_histogram_f32",
  6,
  kAbiArgs_metal_navatala_runtime_deterministic_histogram_f32
};

bool tryGetKernelAbiManifest_metal_runtime(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_runtime_deterministic_reduce_f32") {
    out = &kAbiManifest_metal_navatala_runtime_deterministic_reduce_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_runtime_deterministic_histogram_f32") {
    out = &kAbiManifest_metal_navatala_runtime_deterministic_histogram_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_runtime(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_runtime_deterministic_reduce_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_runtime_deterministic_reduce_f32";
    std::string_view sv(k_metal_navatala_runtime_deterministic_reduce_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_runtime_deterministic_histogram_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_runtime_deterministic_histogram_f32";
    std::string_view sv(k_metal_navatala_runtime_deterministic_histogram_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

