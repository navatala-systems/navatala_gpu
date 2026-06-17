// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_dataframe_top_k` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_dataframe_top_k_init_indices = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_top_k_init_indices(const unsigned int* count, unsigned int* indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    indices[gid] = gid;
  }
}

)kernel";
const char* k_hip_navatala_dataframe_top_k_take_first_k_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_top_k_take_first_k_f32(const float* valuesIn, const unsigned int* indicesIn, const unsigned int* k, float* valuesOut, unsigned int* indicesOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int kVal = k[0];
  bool inBounds = (gid < kVal);
  if (inBounds) {
    float val = valuesIn[gid];
    unsigned int idx = indicesIn[gid];
    valuesOut[gid] = val;
    indicesOut[gid] = idx;
  }
}

)kernel";
const char* k_hip_navatala_dataframe_top_k_take_first_k_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_top_k_take_first_k_f64(const double* valuesIn, const unsigned int* indicesIn, const unsigned int* k, double* valuesOut, unsigned int* indicesOut) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int kVal = k[0];
  bool inBounds = (gid < kVal);
  if (inBounds) {
    double val = valuesIn[gid];
    unsigned int idx = indicesIn[gid];
    valuesOut[gid] = val;
    indicesOut[gid] = idx;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_hip_navatala_dataframe_top_k_init_indices[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_dataframe_top_k_init_indices = {
  1,
  "navatala_dataframe_top_k_init_indices",
  "hip",
  "navatala_dataframe_top_k_init_indices",
  "kernel:hip:navatala_dataframe_top_k_init_indices",
  "abi-r1:hip:navatala_dataframe_top_k_init_indices",
  "abi-r1:hip:navatala_dataframe_top_k_init_indices",
  2,
  kAbiArgs_hip_navatala_dataframe_top_k_init_indices
};

const KernelArgumentInfo kAbiArgs_hip_navatala_dataframe_top_k_take_first_k_f32[] = {
  { "valuesIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indicesIn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "valuesOut", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indicesOut", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_dataframe_top_k_take_first_k_f32 = {
  1,
  "navatala_dataframe_top_k_take_first_k_f32",
  "hip",
  "navatala_dataframe_top_k_take_first_k_f32",
  "kernel:hip:navatala_dataframe_top_k_take_first_k_f32",
  "abi-r1:hip:navatala_dataframe_top_k_take_first_k_f32",
  "abi-r1:hip:navatala_dataframe_top_k_take_first_k_f32",
  5,
  kAbiArgs_hip_navatala_dataframe_top_k_take_first_k_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_dataframe_top_k_take_first_k_f64[] = {
  { "valuesIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indicesIn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "valuesOut", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indicesOut", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_dataframe_top_k_take_first_k_f64 = {
  1,
  "navatala_dataframe_top_k_take_first_k_f64",
  "hip",
  "navatala_dataframe_top_k_take_first_k_f64",
  "kernel:hip:navatala_dataframe_top_k_take_first_k_f64",
  "abi-r1:hip:navatala_dataframe_top_k_take_first_k_f64",
  "abi-r1:hip:navatala_dataframe_top_k_take_first_k_f64",
  5,
  kAbiArgs_hip_navatala_dataframe_top_k_take_first_k_f64
};

bool tryGetKernelAbiManifest_hip_dataframe_top_k(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_dataframe_top_k_init_indices") {
    out = &kAbiManifest_hip_navatala_dataframe_top_k_init_indices;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_top_k_take_first_k_f32") {
    out = &kAbiManifest_hip_navatala_dataframe_top_k_take_first_k_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_top_k_take_first_k_f64") {
    out = &kAbiManifest_hip_navatala_dataframe_top_k_take_first_k_f64;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_dataframe_top_k(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_dataframe_top_k_init_indices") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_dataframe_top_k_init_indices";
    std::string_view sv(k_hip_navatala_dataframe_top_k_init_indices);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_top_k_take_first_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_dataframe_top_k_take_first_k_f32";
    std::string_view sv(k_hip_navatala_dataframe_top_k_take_first_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_top_k_take_first_k_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_dataframe_top_k_take_first_k_f64";
    std::string_view sv(k_hip_navatala_dataframe_top_k_take_first_k_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

