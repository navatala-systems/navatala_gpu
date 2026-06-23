// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_runtime` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_runtime_deterministic_reduce_f32 = R"kernel(
__kernel void navatala_runtime_deterministic_reduce_f32(__global const float* _input, __global const uint* count, __global float* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local float sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float val = _input[gid];
    sdata[lid] = val;
  } else {
    sdata[lid] = as_float(0x00000000u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint f32RedStride = (uint)(128u);
  for (int f32RedStep = 0; f32RedStep < (int)(8); ++f32RedStep) {
    uint f32Stride = f32RedStride;
    if (lid < f32Stride) {
      float other = sdata[(lid + f32Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    uint f32StrideToHalve = f32RedStride;
    uint f32NextStride = (f32StrideToHalve >> (uint)(1u));
    f32RedStride = f32NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    float totalSum = sdata[0];
    result[0] = totalSum;
  }
}

)kernel";
const char* k_opencl_navatala_runtime_deterministic_reduce_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_runtime_deterministic_reduce_f64(__global const double* _input, __global const uint* count, __global double* result) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local double sdata[256];
  uint countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    double val = _input[gid];
    sdata[lid] = val;
  } else {
    sdata[lid] = as_double(0x0000000000000000ul);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  uint f64RedStride = (uint)(128u);
  for (int f64RedStep = 0; f64RedStep < (int)(8); ++f64RedStep) {
    uint f64Stride = f64RedStride;
    if (lid < f64Stride) {
      double other = sdata[(lid + f64Stride)];
      double mine = sdata[lid];
      double sum = (mine + other);
      sdata[lid] = sum;
    }
    uint f64StrideToHalve = f64RedStride;
    uint f64NextStride = (f64StrideToHalve >> (uint)(1u));
    f64RedStride = f64NextStride;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == (uint)(0u)) {
    double totalSum = sdata[0];
    result[0] = totalSum;
  }
}

)kernel";
const char* k_opencl_navatala_runtime_deterministic_histogram_f32 = R"kernel(
__kernel void navatala_runtime_deterministic_histogram_f32(__global const float* values, __global const uint* numBins, __global const float* minVal, __global const float* maxVal, __global const uint* count, __global uint* histogram) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local uint binIndices[256];
  __local uint localHist[256];
  uint countVal = count[0];
  uint numBinsVal = numBins[0];
  float minValF = minVal[0];
  float maxValF = maxVal[0];
  float range = (maxValF - minValF);
  float numBinsFloat = ((float)(numBinsVal));
  float binWidth = (range / numBinsFloat);
  if (lid < numBinsVal) {
    localHist[lid] = (uint)(0u);
  }
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float val = values[gid];
    float normalized = ((val - minValF) / binWidth);
    float binIdxFloat = normalized;
    uint binIdx = ((uint)(binIdxFloat));
    uint maxBinIdx = (numBinsVal - (uint)(1u));
    uint clampedBinIdx = (((binIdx < numBinsVal)) ? (binIdx) : (maxBinIdx));
    binIndices[lid] = clampedBinIdx;
  } else {
    binIndices[lid] = numBinsVal;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < numBinsVal) {
    uint myBin = lid;
    uint histF32RedStride = (uint)(128u);
    for (int histF32RedStep = 0; histF32RedStep < (int)(8); ++histF32RedStep) {
      uint histF32Stride = histF32RedStride;
      if (lid < histF32Stride) {
        uint partnerIdx = (lid + histF32Stride);
        uint storedBin = binIndices[partnerIdx];
        if (storedBin == myBin) {
          uint oldCount = localHist[lid];
          uint newCount = (oldCount + (uint)(1u));
          localHist[lid] = newCount;
        }
      }
      uint histF32StrideToHalve = histF32RedStride;
      uint histF32NextStride = (histF32StrideToHalve >> (uint)(1u));
      histF32RedStride = histF32NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == (uint)(0u)) {
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
const char* k_opencl_navatala_runtime_deterministic_histogram_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_runtime_deterministic_histogram_f64(__global const double* values, __global const uint* numBins, __global const double* minVal, __global const double* maxVal, __global const uint* count, __global uint* histogram) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint lid = ((uint)((int)(get_local_id(0))));
  __local uint binIndices[256];
  __local uint localHist[256];
  uint countVal = count[0];
  uint numBinsVal = numBins[0];
  double minValF = minVal[0];
  double maxValF = maxVal[0];
  double range = (maxValF - minValF);
  double numBinsFloat = ((double)(numBinsVal));
  double binWidth = (range / numBinsFloat);
  if (lid < numBinsVal) {
    localHist[lid] = (uint)(0u);
  }
  bool inBounds = (gid < countVal);
  if (inBounds) {
    double val = values[gid];
    double normalized = ((val - minValF) / binWidth);
    double binIdxFloat = normalized;
    uint binIdx = ((uint)(binIdxFloat));
    uint maxBinIdx = (numBinsVal - (uint)(1u));
    uint clampedBinIdx = (((binIdx < numBinsVal)) ? (binIdx) : (maxBinIdx));
    binIndices[lid] = clampedBinIdx;
  } else {
    binIndices[lid] = numBinsVal;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid < numBinsVal) {
    uint myBin = lid;
    uint histF64RedStride = (uint)(128u);
    for (int histF64RedStep = 0; histF64RedStep < (int)(8); ++histF64RedStep) {
      uint histF64Stride = histF64RedStride;
      if (lid < histF64Stride) {
        uint partnerIdx = (lid + histF64Stride);
        uint storedBin = binIndices[partnerIdx];
        if (storedBin == myBin) {
          uint oldCount = localHist[lid];
          uint newCount = (oldCount + (uint)(1u));
          localHist[lid] = newCount;
        }
      }
      uint histF64StrideToHalve = histF64RedStride;
      uint histF64NextStride = (histF64StrideToHalve >> (uint)(1u));
      histF64RedStride = histF64NextStride;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == (uint)(0u)) {
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

const KernelArgumentInfo kAbiArgs_opencl_navatala_runtime_deterministic_reduce_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_runtime_deterministic_reduce_f32 = {
  1,
  "navatala_runtime_deterministic_reduce_f32",
  "opencl",
  "navatala_runtime_deterministic_reduce_f32",
  "kernel:opencl:navatala_runtime_deterministic_reduce_f32",
  "abi-r1:opencl:navatala_runtime_deterministic_reduce_f32",
  "abi-r1:opencl:navatala_runtime_deterministic_reduce_f32",
  3,
  kAbiArgs_opencl_navatala_runtime_deterministic_reduce_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_runtime_deterministic_reduce_f64[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_runtime_deterministic_reduce_f64 = {
  1,
  "navatala_runtime_deterministic_reduce_f64",
  "opencl",
  "navatala_runtime_deterministic_reduce_f64",
  "kernel:opencl:navatala_runtime_deterministic_reduce_f64",
  "abi-r1:opencl:navatala_runtime_deterministic_reduce_f64",
  "abi-r1:opencl:navatala_runtime_deterministic_reduce_f64",
  3,
  kAbiArgs_opencl_navatala_runtime_deterministic_reduce_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_runtime_deterministic_histogram_f32[] = {
  { "values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBins", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "minVal", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "histogram", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_runtime_deterministic_histogram_f32 = {
  1,
  "navatala_runtime_deterministic_histogram_f32",
  "opencl",
  "navatala_runtime_deterministic_histogram_f32",
  "kernel:opencl:navatala_runtime_deterministic_histogram_f32",
  "abi-r1:opencl:navatala_runtime_deterministic_histogram_f32",
  "abi-r1:opencl:navatala_runtime_deterministic_histogram_f32",
  6,
  kAbiArgs_opencl_navatala_runtime_deterministic_histogram_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_runtime_deterministic_histogram_f64[] = {
  { "values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBins", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "minVal", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "maxVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "histogram", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_runtime_deterministic_histogram_f64 = {
  1,
  "navatala_runtime_deterministic_histogram_f64",
  "opencl",
  "navatala_runtime_deterministic_histogram_f64",
  "kernel:opencl:navatala_runtime_deterministic_histogram_f64",
  "abi-r1:opencl:navatala_runtime_deterministic_histogram_f64",
  "abi-r1:opencl:navatala_runtime_deterministic_histogram_f64",
  6,
  kAbiArgs_opencl_navatala_runtime_deterministic_histogram_f64
};

bool tryGetKernelAbiManifest_opencl_runtime(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_runtime_deterministic_reduce_f32") {
    out = &kAbiManifest_opencl_navatala_runtime_deterministic_reduce_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_runtime_deterministic_reduce_f64") {
    out = &kAbiManifest_opencl_navatala_runtime_deterministic_reduce_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_runtime_deterministic_histogram_f32") {
    out = &kAbiManifest_opencl_navatala_runtime_deterministic_histogram_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_runtime_deterministic_histogram_f64") {
    out = &kAbiManifest_opencl_navatala_runtime_deterministic_histogram_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_runtime(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_runtime_deterministic_reduce_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_runtime_deterministic_reduce_f32";
    std::string_view sv(k_opencl_navatala_runtime_deterministic_reduce_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_runtime_deterministic_reduce_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_runtime_deterministic_reduce_f64";
    std::string_view sv(k_opencl_navatala_runtime_deterministic_reduce_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_runtime_deterministic_histogram_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_runtime_deterministic_histogram_f32";
    std::string_view sv(k_opencl_navatala_runtime_deterministic_histogram_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_runtime_deterministic_histogram_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_runtime_deterministic_histogram_f64";
    std::string_view sv(k_opencl_navatala_runtime_deterministic_histogram_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

