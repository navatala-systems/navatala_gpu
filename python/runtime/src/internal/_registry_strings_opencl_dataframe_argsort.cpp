// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `opencl_dataframe_argsort` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_opencl_navatala_dataframe_argsort_init_indices = R"kernel(
__kernel void navatala_dataframe_argsort_init_indices(__global const uint* count, __global uint* indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  bool inBounds = (gid < n);
  if (inBounds) {
    indices[gid] = gid;
  }
}

)kernel";
const char* k_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i32 = R"kernel(
__kernel void navatala_dataframe_argsort_bitonic_step_argsort_i32(__global const int* keys, __global const uint* count, __global const uint* stepSize, __global const uint* halfBlockSize, __global uint* indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint step = stepSize[(uint)(0u)];
  uint halfBlock = halfBlockSize[(uint)(0u)];
  uint pairIdx = (gid * (uint)(2u));
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
    uint blockPairIdx = (i / (step * (uint)(2u)));
    bool ascending = ((blockPairIdx % (uint)(2u)) == (uint)(0u));
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
const char* k_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f32 = R"kernel(
__kernel void navatala_dataframe_argsort_bitonic_step_argsort_f32(__global const float* keys, __global const uint* count, __global const uint* stepSize, __global const uint* halfBlockSize, __global uint* indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint step = stepSize[(uint)(0u)];
  uint halfBlock = halfBlockSize[(uint)(0u)];
  uint pairIdx = (gid * (uint)(2u));
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
    uint blockPairIdx = (i / (step * (uint)(2u)));
    bool ascending = ((blockPairIdx % (uint)(2u)) == (uint)(0u));
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
const char* k_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i64 = R"kernel(
__kernel void navatala_dataframe_argsort_bitonic_step_argsort_i64(__global const long* keys, __global const uint* count, __global const uint* stepSize, __global const uint* halfBlockSize, __global uint* indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint step = stepSize[(uint)(0u)];
  uint halfBlock = halfBlockSize[(uint)(0u)];
  uint pairIdx = (gid * (uint)(2u));
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
    uint blockPairIdx = (i / (step * (uint)(2u)));
    bool ascending = ((blockPairIdx % (uint)(2u)) == (uint)(0u));
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
const char* k_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f64 = R"kernel(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void navatala_dataframe_argsort_bitonic_step_argsort_f64(__global const double* keys, __global const uint* count, __global const uint* stepSize, __global const uint* halfBlockSize, __global uint* indices) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint step = stepSize[(uint)(0u)];
  uint halfBlock = halfBlockSize[(uint)(0u)];
  uint pairIdx = (gid * (uint)(2u));
  uint blockIdx = (pairIdx / step);
  uint inBlock = (pairIdx % step);
  uint i = ((blockIdx * step) + inBlock);
  uint j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    uint idxI = indices[i];
    uint idxJ = indices[j];
    double keyI = keys[idxI];
    double keyJ = keys[idxJ];
    uint blockPairIdx = (i / (step * (uint)(2u)));
    bool ascending = ((blockPairIdx % (uint)(2u)) == (uint)(0u));
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
const char* k_opencl_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32 = R"kernel(
__kernel void navatala_dataframe_argsort_bitonic_step_sort_by_key_i32(__global const uint* count, __global const uint* stepSize, __global const uint* halfBlockSize, __global int* keys, __global int* values) {
  int gid0 = (int)get_global_id(0);
  uint gid = ((uint)((int)(get_global_id(0))));
  uint n = count[(uint)(0u)];
  uint step = stepSize[(uint)(0u)];
  uint halfBlock = halfBlockSize[(uint)(0u)];
  uint pairIdx = (gid * (uint)(2u));
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
    uint blockPairIdx = (i / (step * (uint)(2u)));
    bool ascending = ((blockPairIdx % (uint)(2u)) == (uint)(0u));
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

const KernelArgumentInfo kAbiArgs_opencl_navatala_dataframe_argsort_init_indices[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_dataframe_argsort_init_indices = {
  1,
  "navatala_dataframe_argsort_init_indices",
  "opencl",
  "navatala_dataframe_argsort_init_indices",
  "kernel:opencl:navatala_dataframe_argsort_init_indices",
  "abi-r1:opencl:navatala_dataframe_argsort_init_indices",
  "abi-r1:opencl:navatala_dataframe_argsort_init_indices",
  2,
  kAbiArgs_opencl_navatala_dataframe_argsort_init_indices
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i32[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i32 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "opencl",
  "navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "kernel:opencl:navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_argsort_i32",
  5,
  kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f32[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f32 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "opencl",
  "navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "kernel:opencl:navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_argsort_f32",
  5,
  kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f32
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i64[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i64 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "opencl",
  "navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "kernel:opencl:navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_argsort_i64",
  5,
  kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f64[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f64 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_f64",
  "opencl",
  "navatala_dataframe_argsort_bitonic_step_argsort_f64",
  "kernel:opencl:navatala_dataframe_argsort_bitonic_step_argsort_f64",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_argsort_f64",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_argsort_f64",
  5,
  kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f64
};

const KernelArgumentInfo kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "keys", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "opencl",
  "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "kernel:opencl:navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "abi-r1:opencl:navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  5,
  kAbiArgs_opencl_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32
};

bool tryGetKernelAbiManifest_opencl_dataframe_argsort(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_init_indices") {
    out = &kAbiManifest_opencl_navatala_dataframe_argsort_init_indices;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i32") {
    out = &kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f32") {
    out = &kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f32;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i64") {
    out = &kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f64") {
    out = &kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f64;
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32") {
    out = &kAbiManifest_opencl_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_opencl_dataframe_argsort(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_init_indices") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_dataframe_argsort_init_indices";
    std::string_view sv(k_opencl_navatala_dataframe_argsort_init_indices);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_i32";
    std::string_view sv(k_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_f32";
    std::string_view sv(k_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_i64";
    std::string_view sv(k_opencl_navatala_dataframe_argsort_bitonic_step_argsort_i64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_f64";
    std::string_view sv(k_opencl_navatala_dataframe_argsort_bitonic_step_argsort_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "opencl" && kernelName == "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::OpenClC;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32";
    std::string_view sv(k_opencl_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

