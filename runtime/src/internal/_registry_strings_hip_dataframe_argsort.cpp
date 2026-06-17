// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `hip_dataframe_argsort` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_hip_navatala_dataframe_argsort_init_indices = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_argsort_init_indices(const unsigned int* count, unsigned int* indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  bool inBounds = (gid < n);
  if (inBounds) {
    indices[gid] = gid;
  }
}

)kernel";
const char* k_hip_navatala_dataframe_argsort_bitonic_step_argsort_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_argsort_bitonic_step_argsort_i32(const int* keys, const unsigned int* count, const unsigned int* stepSize, const unsigned int* halfBlockSize, unsigned int* indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int step = stepSize[0u];
  unsigned int halfBlock = halfBlockSize[0u];
  unsigned int pairIdx = (gid * 2u);
  unsigned int blockIdx = (pairIdx / step);
  unsigned int inBlock = (pairIdx % step);
  unsigned int i = ((blockIdx * step) + inBlock);
  unsigned int j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    unsigned int idxI = indices[i];
    unsigned int idxJ = indices[j];
    int keyI = keys[idxI];
    int keyJ = keys[idxJ];
    unsigned int blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    unsigned int newIdxI = ((needSwap) ? (idxJ) : (idxI));
    unsigned int newIdxJ = ((needSwap) ? (idxI) : (idxJ));
    indices[i] = newIdxI;
    indices[j] = newIdxJ;
  }
}

)kernel";
const char* k_hip_navatala_dataframe_argsort_bitonic_step_argsort_f32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_argsort_bitonic_step_argsort_f32(const float* keys, const unsigned int* count, const unsigned int* stepSize, const unsigned int* halfBlockSize, unsigned int* indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int step = stepSize[0u];
  unsigned int halfBlock = halfBlockSize[0u];
  unsigned int pairIdx = (gid * 2u);
  unsigned int blockIdx = (pairIdx / step);
  unsigned int inBlock = (pairIdx % step);
  unsigned int i = ((blockIdx * step) + inBlock);
  unsigned int j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    unsigned int idxI = indices[i];
    unsigned int idxJ = indices[j];
    float keyI = keys[idxI];
    float keyJ = keys[idxJ];
    unsigned int blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    unsigned int newIdxI = ((needSwap) ? (idxJ) : (idxI));
    unsigned int newIdxJ = ((needSwap) ? (idxI) : (idxJ));
    indices[i] = newIdxI;
    indices[j] = newIdxJ;
  }
}

)kernel";
const char* k_hip_navatala_dataframe_argsort_bitonic_step_argsort_i64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_argsort_bitonic_step_argsort_i64(const long long* keys, const unsigned int* count, const unsigned int* stepSize, const unsigned int* halfBlockSize, unsigned int* indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int step = stepSize[0u];
  unsigned int halfBlock = halfBlockSize[0u];
  unsigned int pairIdx = (gid * 2u);
  unsigned int blockIdx = (pairIdx / step);
  unsigned int inBlock = (pairIdx % step);
  unsigned int i = ((blockIdx * step) + inBlock);
  unsigned int j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    unsigned int idxI = indices[i];
    unsigned int idxJ = indices[j];
    long long keyI = keys[idxI];
    long long keyJ = keys[idxJ];
    unsigned int blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    unsigned int newIdxI = ((needSwap) ? (idxJ) : (idxI));
    unsigned int newIdxJ = ((needSwap) ? (idxI) : (idxJ));
    indices[i] = newIdxI;
    indices[j] = newIdxJ;
  }
}

)kernel";
const char* k_hip_navatala_dataframe_argsort_bitonic_step_argsort_f64 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_argsort_bitonic_step_argsort_f64(const double* keys, const unsigned int* count, const unsigned int* stepSize, const unsigned int* halfBlockSize, unsigned int* indices) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int step = stepSize[0u];
  unsigned int halfBlock = halfBlockSize[0u];
  unsigned int pairIdx = (gid * 2u);
  unsigned int blockIdx = (pairIdx / step);
  unsigned int inBlock = (pairIdx % step);
  unsigned int i = ((blockIdx * step) + inBlock);
  unsigned int j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    unsigned int idxI = indices[i];
    unsigned int idxJ = indices[j];
    double keyI = keys[idxI];
    double keyJ = keys[idxJ];
    unsigned int blockPairIdx = (i / (step * 2u));
    bool ascending = ((blockPairIdx % 2u) == 0u);
    bool needSwapAsc = (keyI > keyJ);
    bool needSwapDesc = (keyI < keyJ);
    bool needSwap = ((ascending) ? (needSwapAsc) : (needSwapDesc));
    unsigned int newIdxI = ((needSwap) ? (idxJ) : (idxI));
    unsigned int newIdxJ = ((needSwap) ? (idxI) : (idxJ));
    indices[i] = newIdxI;
    indices[j] = newIdxJ;
  }
}

)kernel";
const char* k_hip_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32 = R"kernel(
#include <hip/hip_runtime.h>
extern "C" __global__ void navatala_dataframe_argsort_bitonic_step_sort_by_key_i32(const unsigned int* count, const unsigned int* stepSize, const unsigned int* halfBlockSize, int* keys, int* values) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int gid = ((unsigned int)((int)(blockIdx.x * blockDim.x + threadIdx.x)));
  unsigned int n = count[0u];
  unsigned int step = stepSize[0u];
  unsigned int halfBlock = halfBlockSize[0u];
  unsigned int pairIdx = (gid * 2u);
  unsigned int blockIdx = (pairIdx / step);
  unsigned int inBlock = (pairIdx % step);
  unsigned int i = ((blockIdx * step) + inBlock);
  unsigned int j = (i + halfBlock);
  bool inBounds = (j < n);
  if (inBounds) {
    int keyI = keys[i];
    int keyJ = keys[j];
    int valI = values[i];
    int valJ = values[j];
    unsigned int blockPairIdx = (i / (step * 2u));
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

const KernelArgumentInfo kAbiArgs_hip_navatala_dataframe_argsort_init_indices[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 1, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_dataframe_argsort_init_indices = {
  1,
  "navatala_dataframe_argsort_init_indices",
  "hip",
  "navatala_dataframe_argsort_init_indices",
  "kernel:hip:navatala_dataframe_argsort_init_indices",
  "abi-r1:hip:navatala_dataframe_argsort_init_indices",
  "abi-r1:hip:navatala_dataframe_argsort_init_indices",
  2,
  kAbiArgs_hip_navatala_dataframe_argsort_init_indices
};

const KernelArgumentInfo kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_argsort_i32[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_argsort_i32 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "hip",
  "navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "kernel:hip:navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_argsort_i32",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_argsort_i32",
  5,
  kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_argsort_i32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_argsort_f32[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_argsort_f32 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "hip",
  "navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "kernel:hip:navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_argsort_f32",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_argsort_f32",
  5,
  kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_argsort_f32
};

const KernelArgumentInfo kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_argsort_i64[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_argsort_i64 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "hip",
  "navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "kernel:hip:navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_argsort_i64",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_argsort_i64",
  5,
  kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_argsort_i64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_argsort_f64[] = {
  { "keys", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "count", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 3, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "indices", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_argsort_f64 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_argsort_f64",
  "hip",
  "navatala_dataframe_argsort_bitonic_step_argsort_f64",
  "kernel:hip:navatala_dataframe_argsort_bitonic_step_argsort_f64",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_argsort_f64",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_argsort_f64",
  5,
  kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_argsort_f64
};

const KernelArgumentInfo kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32[] = {
  { "count", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "stepSize", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "halfBlockSize", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "keys", 3, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "values", 4, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32 = {
  1,
  "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "hip",
  "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "kernel:hip:navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  "abi-r1:hip:navatala_dataframe_argsort_bitonic_step_sort_by_key_i32",
  5,
  kAbiArgs_hip_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32
};

bool tryGetKernelAbiManifest_hip_dataframe_argsort(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_init_indices") {
    out = &kAbiManifest_hip_navatala_dataframe_argsort_init_indices;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i32") {
    out = &kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_argsort_i32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f32") {
    out = &kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_argsort_f32;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i64") {
    out = &kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_argsort_i64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f64") {
    out = &kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_argsort_f64;
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32") {
    out = &kAbiManifest_hip_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_hip_dataframe_argsort(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_init_indices") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_dataframe_argsort_init_indices";
    std::string_view sv(k_hip_navatala_dataframe_argsort_init_indices);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_i32";
    std::string_view sv(k_hip_navatala_dataframe_argsort_bitonic_step_argsort_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_f32";
    std::string_view sv(k_hip_navatala_dataframe_argsort_bitonic_step_argsort_f32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_i64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_i64";
    std::string_view sv(k_hip_navatala_dataframe_argsort_bitonic_step_argsort_i64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_argsort_f64") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_argsort_f64";
    std::string_view sv(k_hip_navatala_dataframe_argsort_bitonic_step_argsort_f64);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "hip" && kernelName == "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32") {
    out.kind = GpuRuntime::ProgramSource::Kind::HipCpp;
    out.entryPoint = "navatala_dataframe_argsort_bitonic_step_sort_by_key_i32";
    std::string_view sv(k_hip_navatala_dataframe_argsort_bitonic_step_sort_by_key_i32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

