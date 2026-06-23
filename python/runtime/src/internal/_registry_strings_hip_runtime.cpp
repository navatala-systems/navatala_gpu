// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_runtime` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_runtime_deterministic_reduce_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_runtime_deterministic_reduce_f32(const float* _input, const unsigned int* count, float* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ float sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    float val = _input[gid];
    sdata[lid] = val;
  } else {
    sdata[lid] = __uint_as_float(0x00000000u);
  }
  __syncthreads();
  unsigned int f32RedStride = 128u;
  for (int f32RedStep = 0; f32RedStep < (int)(8); ++f32RedStep) {
    unsigned int f32Stride = f32RedStride;
    if (lid < f32Stride) {
      float other = sdata[(lid + f32Stride)];
      float mine = sdata[lid];
      float sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int f32StrideToHalve = f32RedStride;
    unsigned int f32NextStride = (f32StrideToHalve >> 1u);
    f32RedStride = f32NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    float totalSum = sdata[0];
    result[0] = totalSum;
  }
}

)kernel";
const char* k_hip_navatala_runtime_deterministic_reduce_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_runtime_deterministic_reduce_f64(const double* _input, const unsigned int* count, double* result) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ double sdata[256];
  unsigned int countVal = count[0];
  bool inBounds = (gid < countVal);
  if (inBounds) {
    double val = _input[gid];
    sdata[lid] = val;
  } else {
    sdata[lid] = __longlong_as_double(0x0000000000000000ull);
  }
  __syncthreads();
  unsigned int f64RedStride = 128u;
  for (int f64RedStep = 0; f64RedStep < (int)(8); ++f64RedStep) {
    unsigned int f64Stride = f64RedStride;
    if (lid < f64Stride) {
      double other = sdata[(lid + f64Stride)];
      double mine = sdata[lid];
      double sum = (mine + other);
      sdata[lid] = sum;
    }
    unsigned int f64StrideToHalve = f64RedStride;
    unsigned int f64NextStride = (f64StrideToHalve >> 1u);
    f64RedStride = f64NextStride;
    __syncthreads();
  }
  if (lid == 0u) {
    double totalSum = sdata[0];
    result[0] = totalSum;
  }
}

)kernel";
const char* k_hip_navatala_runtime_deterministic_histogram_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_runtime_deterministic_histogram_f32(const float* values, const unsigned int* numBins, const float* minVal, const float* maxVal, const unsigned int* count, unsigned int* histogram) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ unsigned int binIndices[256];
  __shared__ unsigned int localHist[256];
  unsigned int countVal = count[0];
  unsigned int numBinsVal = numBins[0];
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
    unsigned int binIdx = ((unsigned int)(binIdxFloat));
    unsigned int maxBinIdx = (numBinsVal - 1u);
    unsigned int clampedBinIdx = (((binIdx < numBinsVal)) ? (binIdx) : (maxBinIdx));
    binIndices[lid] = clampedBinIdx;
  } else {
    binIndices[lid] = numBinsVal;
  }
  __syncthreads();
  if (lid < numBinsVal) {
    unsigned int myBin = lid;
    unsigned int histF32RedStride = 128u;
    for (int histF32RedStep = 0; histF32RedStep < (int)(8); ++histF32RedStep) {
      unsigned int histF32Stride = histF32RedStride;
      if (lid < histF32Stride) {
        unsigned int partnerIdx = (lid + histF32Stride);
        unsigned int storedBin = binIndices[partnerIdx];
        if (storedBin == myBin) {
          unsigned int oldCount = localHist[lid];
          unsigned int newCount = (oldCount + 1u);
          localHist[lid] = newCount;
        }
      }
      unsigned int histF32StrideToHalve = histF32RedStride;
      unsigned int histF32NextStride = (histF32StrideToHalve >> 1u);
      histF32RedStride = histF32NextStride;
      __syncthreads();
    }
  }
  __syncthreads();
  if (lid == 0u) {
    for (int bin = 0; bin < (int)(numBinsVal); ++bin) {
      unsigned int binU32 = ((unsigned int)(bin));
      unsigned int localCount = localHist[binU32];
      unsigned int globalCount = histogram[binU32];
      unsigned int newGlobalCount = (globalCount + localCount);
      histogram[binU32] = newGlobalCount;
    }
  }
}

)kernel";
const char* k_hip_navatala_runtime_deterministic_histogram_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_runtime_deterministic_histogram_f64(const double* values, const unsigned int* numBins, const double* minVal, const double* maxVal, const unsigned int* count, unsigned int* histogram) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int lid = ((unsigned int)((int)(threadIdx.x)));
  __shared__ unsigned int binIndices[256];
  __shared__ unsigned int localHist[256];
  unsigned int countVal = count[0];
  unsigned int numBinsVal = numBins[0];
  double minValF = minVal[0];
  double maxValF = maxVal[0];
  double range = (maxValF - minValF);
  double numBinsFloat = ((double)(numBinsVal));
  double binWidth = (range / numBinsFloat);
  if (lid < numBinsVal) {
    localHist[lid] = 0u;
  }
  bool inBounds = (gid < countVal);
  if (inBounds) {
    double val = values[gid];
    double normalized = ((val - minValF) / binWidth);
    double binIdxFloat = normalized;
    unsigned int binIdx = ((unsigned int)(binIdxFloat));
    unsigned int maxBinIdx = (numBinsVal - 1u);
    unsigned int clampedBinIdx = (((binIdx < numBinsVal)) ? (binIdx) : (maxBinIdx));
    binIndices[lid] = clampedBinIdx;
  } else {
    binIndices[lid] = numBinsVal;
  }
  __syncthreads();
  if (lid < numBinsVal) {
    unsigned int myBin = lid;
    unsigned int histF64RedStride = 128u;
    for (int histF64RedStep = 0; histF64RedStep < (int)(8); ++histF64RedStep) {
      unsigned int histF64Stride = histF64RedStride;
      if (lid < histF64Stride) {
        unsigned int partnerIdx = (lid + histF64Stride);
        unsigned int storedBin = binIndices[partnerIdx];
        if (storedBin == myBin) {
          unsigned int oldCount = localHist[lid];
          unsigned int newCount = (oldCount + 1u);
          localHist[lid] = newCount;
        }
      }
      unsigned int histF64StrideToHalve = histF64RedStride;
      unsigned int histF64NextStride = (histF64StrideToHalve >> 1u);
      histF64RedStride = histF64NextStride;
      __syncthreads();
    }
  }
  __syncthreads();
  if (lid == 0u) {
    for (int bin = 0; bin < (int)(numBinsVal); ++bin) {
      unsigned int binU32 = ((unsigned int)(bin));
      unsigned int localCount = localHist[binU32];
      unsigned int globalCount = histogram[binU32];
      unsigned int newGlobalCount = (globalCount + localCount);
      histogram[binU32] = newGlobalCount;
    }
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_runtime_deterministic_reduce_f32[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_runtime_deterministic_reduce_f32 = {
  1,
  "navatala_runtime_deterministic_reduce_f32",
  "hip",
  "navatala_runtime_deterministic_reduce_f32",
  "kernel:hip:navatala_runtime_deterministic_reduce_f32",
  "abi-r1:hip:navatala_runtime_deterministic_reduce_f32",
  "abi-r1:hip:navatala_runtime_deterministic_reduce_f32",
  3,
  kAbiArgs_hip_navatala_runtime_deterministic_reduce_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_runtime_deterministic_reduce_f64[] = {
  { "input", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "result", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_runtime_deterministic_reduce_f64 = {
  1,
  "navatala_runtime_deterministic_reduce_f64",
  "hip",
  "navatala_runtime_deterministic_reduce_f64",
  "kernel:hip:navatala_runtime_deterministic_reduce_f64",
  "abi-r1:hip:navatala_runtime_deterministic_reduce_f64",
  "abi-r1:hip:navatala_runtime_deterministic_reduce_f64",
  3,
  kAbiArgs_hip_navatala_runtime_deterministic_reduce_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_runtime_deterministic_histogram_f32[] = {
  { "values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBins", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "minVal", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "maxVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "histogram", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_runtime_deterministic_histogram_f32 = {
  1,
  "navatala_runtime_deterministic_histogram_f32",
  "hip",
  "navatala_runtime_deterministic_histogram_f32",
  "kernel:hip:navatala_runtime_deterministic_histogram_f32",
  "abi-r1:hip:navatala_runtime_deterministic_histogram_f32",
  "abi-r1:hip:navatala_runtime_deterministic_histogram_f32",
  6,
  kAbiArgs_hip_navatala_runtime_deterministic_histogram_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_runtime_deterministic_histogram_f64[] = {
  { "values", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "numBins", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "minVal", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "maxVal", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 8, 8, 256, nullptr, 0, 0 },
  { "count", 4, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "histogram", 5, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_runtime_deterministic_histogram_f64 = {
  1,
  "navatala_runtime_deterministic_histogram_f64",
  "hip",
  "navatala_runtime_deterministic_histogram_f64",
  "kernel:hip:navatala_runtime_deterministic_histogram_f64",
  "abi-r1:hip:navatala_runtime_deterministic_histogram_f64",
  "abi-r1:hip:navatala_runtime_deterministic_histogram_f64",
  6,
  kAbiArgs_hip_navatala_runtime_deterministic_histogram_f64
};

bool tryGetKernelAbiManifest_hip_runtime(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_runtime_deterministic_reduce_f32") {
    out = &kAbiManifest_hip_navatala_runtime_deterministic_reduce_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_runtime_deterministic_reduce_f64") {
    out = &kAbiManifest_hip_navatala_runtime_deterministic_reduce_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_runtime_deterministic_histogram_f32") {
    out = &kAbiManifest_hip_navatala_runtime_deterministic_histogram_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_runtime_deterministic_histogram_f64") {
    out = &kAbiManifest_hip_navatala_runtime_deterministic_histogram_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_runtime(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_runtime_deterministic_reduce_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_runtime_deterministic_reduce_f32";
    std::string_view sv(k_hip_navatala_runtime_deterministic_reduce_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_runtime_deterministic_reduce_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_runtime_deterministic_reduce_f64";
    std::string_view sv(k_hip_navatala_runtime_deterministic_reduce_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_runtime_deterministic_histogram_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_runtime_deterministic_histogram_f32";
    std::string_view sv(k_hip_navatala_runtime_deterministic_histogram_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_runtime_deterministic_histogram_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_runtime_deterministic_histogram_f64";
    std::string_view sv(k_hip_navatala_runtime_deterministic_histogram_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

