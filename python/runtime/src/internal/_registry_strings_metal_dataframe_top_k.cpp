// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_dataframe_top_k` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_dataframe_top_k_init_indices = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_top_k_init_indices(device const uint* count [[buffer(0)]], device uint* indices [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0];
  bool inBounds = (gid < n);
  if (inBounds) {
    indices[gid] = gid;
  }
}

)kernel";
const char* k_metal_navatala_dataframe_top_k_take_first_k_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_top_k_take_first_k_f32(device const float* valuesIn [[buffer(0)]], device const uint* indicesIn [[buffer(1)]], device const uint* k [[buffer(2)]], device float* valuesOut [[buffer(3)]], device uint* indicesOut [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint kVal = k[0];
  bool inBounds = (gid < kVal);
  if (inBounds) {
    float val = valuesIn[gid];
    uint idx = indicesIn[gid];
    valuesOut[gid] = val;
    indicesOut[gid] = idx;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_dataframe_top_k_init_indices[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_dataframe_top_k_init_indices = {
  1,
  "navatala_dataframe_top_k_init_indices",
  "metal",
  "navatala_dataframe_top_k_init_indices",
  "kernel:metal:navatala_dataframe_top_k_init_indices",
  "abi-r1:metal:navatala_dataframe_top_k_init_indices",
  "abi-r1:metal:navatala_dataframe_top_k_init_indices",
  2,
  kAbiArgs_metal_navatala_dataframe_top_k_init_indices
};

const KernelArgumentInfo kAbiArgs_metal_navatala_dataframe_top_k_take_first_k_f32[] = {
  { "valuesIn", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indicesIn", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "k", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "valuesOut", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "indicesOut", 4, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_dataframe_top_k_take_first_k_f32 = {
  1,
  "navatala_dataframe_top_k_take_first_k_f32",
  "metal",
  "navatala_dataframe_top_k_take_first_k_f32",
  "kernel:metal:navatala_dataframe_top_k_take_first_k_f32",
  "abi-r1:metal:navatala_dataframe_top_k_take_first_k_f32",
  "abi-r1:metal:navatala_dataframe_top_k_take_first_k_f32",
  5,
  kAbiArgs_metal_navatala_dataframe_top_k_take_first_k_f32
};

bool tryGetKernelAbiManifest_metal_dataframe_top_k(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_dataframe_top_k_init_indices") {
    out = &kAbiManifest_metal_navatala_dataframe_top_k_init_indices;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_top_k_take_first_k_f32") {
    out = &kAbiManifest_metal_navatala_dataframe_top_k_take_first_k_f32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_dataframe_top_k(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_dataframe_top_k_init_indices") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_dataframe_top_k_init_indices";
    std::string_view sv(k_metal_navatala_dataframe_top_k_init_indices);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_top_k_take_first_k_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_dataframe_top_k_take_first_k_f32";
    std::string_view sv(k_metal_navatala_dataframe_top_k_take_first_k_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

