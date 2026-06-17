// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `metal_dataframe_argsort` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_metal_navatala_dataframe_argsort_init_indices = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_argsort_init_indices(device const uint* count [[buffer(0)]], device uint* indices [[buffer(1)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    indices[gid] = gid;
  }
}

)kernel";
const char* k_metal_navatala_dataframe_argsort_bitonic_step_argsort_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_argsort_bitonic_step_argsort_i32(device const int* keys [[buffer(0)]], device const uint* count [[buffer(1)]], device const uint* stepSize [[buffer(2)]], device const uint* halfBlockSize [[buffer(3)]], device uint* indices [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint step = stepSize[0u];
  uint halfBlock = halfBlockSize[0u];
  uint pairIdx = (gid * 2u);
  uint blockIdx = (pairIdx / step);
  uint inBlock = (pairIdx % step);
  uint i = ((blockIdx * step) + inBlock);
  uint j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    uint idxI = indices[i];
    uint idxJ = indices[j];
    int keyI = keys[idxI];
    int keyJ = keys[idxJ];
    uint blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    uint newIdxI = ((needSwap) ? (idxJ) : (idxI));
    uint newIdxJ = ((needSwap) ? (idxI) : (idxJ));
    indices[i] = newIdxI;
    indices[j] = newIdxJ;
  }
}

)kernel";
const char* k_metal_navatala_dataframe_argsort_bitonic_step_argsort_f32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_argsort_bitonic_step_argsort_f32(device const float* keys [[buffer(0)]], device const uint* count [[buffer(1)]], device const uint* stepSize [[buffer(2)]], device const uint* halfBlockSize [[buffer(3)]], device uint* indices [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint step = stepSize[0u];
  uint halfBlock = halfBlockSize[0u];
  uint pairIdx = (gid * 2u);
  uint blockIdx = (pairIdx / step);
  uint inBlock = (pairIdx % step);
  uint i = ((blockIdx * step) + inBlock);
  uint j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    uint idxI = indices[i];
    uint idxJ = indices[j];
    float keyI = keys[idxI];
    float keyJ = keys[idxJ];
    uint blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    uint newIdxI = ((needSwap) ? (idxJ) : (idxI));
    uint newIdxJ = ((needSwap) ? (idxI) : (idxJ));
    indices[i] = newIdxI;
    indices[j] = newIdxJ;
  }
}

)kernel";
const char* k_metal_navatala_dataframe_argsort_bitonic_step_argsort_i64 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_argsort_bitonic_step_argsort_i64(device const long* keys [[buffer(0)]], device const uint* count [[buffer(1)]], device const uint* stepSize [[buffer(2)]], device const uint* halfBlockSize [[buffer(3)]], device uint* indices [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint step = stepSize[0u];
  uint halfBlock = halfBlockSize[0u];
  uint pairIdx = (gid * 2u);
  uint blockIdx = (pairIdx / step);
  uint inBlock = (pairIdx % step);
  uint i = ((blockIdx * step) + inBlock);
  uint j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    uint idxI = indices[i];
    uint idxJ = indices[j];
    long keyI = keys[idxI];
    long keyJ = keys[idxJ];
    uint blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    uint newIdxI = ((needSwap) ? (idxJ) : (idxI));
    uint newIdxJ = ((needSwap) ? (idxI) : (idxJ));
    indices[i] = newIdxI;
    indices[j] = newIdxJ;
  }
}

)kernel";
const char* k_metal_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32 = R"kernel(
#include <metal_stdlib>
using namespace metal;

kernel void navatala_dataframe_argsort_bitonic_step_sort_by_key_i32(device const uint* count [[buffer(0)]], device const uint* stepSize [[buffer(1)]], device const uint* halfBlockSize [[buffer(2)]], device int* keys [[buffer(3)]], device int* values [[buffer(4)]], uint3 __gid [[thread_position_in_grid]], uint3 __tid [[thread_position_in_threadgroup]], uint3 __tgid [[threadgroup_position_in_grid]], uint3 __tgsz [[threads_per_threadgroup]], uint3 __grid_size [[threads_per_grid]], uint __lane [[thread_index_in_simdgroup]], uint __simd_size [[threads_per_simdgroup]]) {
  uint gid = ((uint)(int(__gid.x)));
  uint n = count[0u];
  uint step = stepSize[0u];
  uint halfBlock = halfBlockSize[0u];
  uint pairIdx = (gid * 2u);
  uint blockIdx = (pairIdx / step);
  uint inBlock = (pairIdx % step);
  uint i = ((blockIdx * step) + inBlock);
  uint j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    int keyI = keys[i];
    int keyJ = keys[j];
    int valI = values[i];
    int valJ = values[j];
    uint blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    int newKeyI = ((needSwap) ? (keyJ) : (keyI));
    int newKeyJ = ((needSwap) ? (keyI) : (keyJ));
    int newValI = ((needSwap) ? (valJ) : (valI));
    int newValJ = ((needSwap) ? (valI) : (valJ));
    keys[i] = newKeyI;
    keys[j] = newKeyJ;
    values[i] = newValI;
    values[j] = newValJ;
  }
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_metal_navatala_dataframe_argsort_init_indices[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_dataframe_argsort_init_indices = {
  1,
  "navatala_dataframe_argsort_init_indices",
  "metal",
  "navatala_dataframe_argsort_init_indices",
  "kernel:metal:navatala_dataframe_argsort_init_indices",
  "abi-r1:metal:navatala_dataframe_argsort_init_indices",
  "abi-r1:metal:navatala_dataframe_argsort_init_indices",
  2,
  kAbiArgs_metal_navatala_dataframe_argsort_init_indices
};

const KernelArgumentInfo kAbiArgs_metal_navatala_dataframe_argsort_bitonic_step_argsort_i32[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_dataframe_argsort_bitonic_step_argsort_i32 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "metal",
  "navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "kernel:metal:navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "abi-r1:metal:navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "abi-r1:metal:navatala_dataframe_argsort_bitonic_step_argsort_i32",
  5,
  kAbiArgs_metal_navatala_dataframe_argsort_bitonic_step_argsort_i32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_dataframe_argsort_bitonic_step_argsort_f32[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_dataframe_argsort_bitonic_step_argsort_f32 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "metal",
  "navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "kernel:metal:navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "abi-r1:metal:navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "abi-r1:metal:navatala_dataframe_argsort_bitonic_step_argsort_f32",
  5,
  kAbiArgs_metal_navatala_dataframe_argsort_bitonic_step_argsort_f32
};

const KernelArgumentInfo kAbiArgs_metal_navatala_dataframe_argsort_bitonic_step_argsort_i64[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_dataframe_argsort_bitonic_step_argsort_i64 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "metal",
  "navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "kernel:metal:navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "abi-r1:metal:navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "abi-r1:metal:navatala_dataframe_argsort_bitonic_step_argsort_i64",
  5,
  kAbiArgs_metal_navatala_dataframe_argsort_bitonic_step_argsort_i64
};

const KernelArgumentInfo kAbiArgs_metal_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "keys", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_metal_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "metal",
  "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "kernel:metal:navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "abi-r1:metal:navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "abi-r1:metal:navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  5,
  kAbiArgs_metal_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32
};

bool tryGetKernelAbiManifest_metal_dataframe_argsort(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_init_indices") {
    out = &kAbiManifest_metal_navatala_dataframe_argsort_init_indices;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i32") {
    out = &kAbiManifest_metal_navatala_dataframe_argsort_bitonic_step_argsort_i32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f32") {
    out = &kAbiManifest_metal_navatala_dataframe_argsort_bitonic_step_argsort_f32;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i64") {
    out = &kAbiManifest_metal_navatala_dataframe_argsort_bitonic_step_argsort_i64;
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32") {
    out = &kAbiManifest_metal_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_metal_dataframe_argsort(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_init_indices") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_dataframe_argsort_init_indices";
    std::string_view sv(k_metal_navatala_dataframe_argsort_init_indices);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_i32";
    std::string_view sv(k_metal_navatala_dataframe_argsort_bitonic_step_argsort_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_f32";
    std::string_view sv(k_metal_navatala_dataframe_argsort_bitonic_step_argsort_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i64") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_i64";
    std::string_view sv(k_metal_navatala_dataframe_argsort_bitonic_step_argsort_i64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "metal" && kernelName == "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::Msl;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32";
    std::string_view sv(k_metal_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

