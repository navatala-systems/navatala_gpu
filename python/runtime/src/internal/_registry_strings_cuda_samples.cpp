// SPDX-License-Identifier: Apache-2.0
// Kernel registry partition for the `cuda_samples` bucket. Compiles
// against the declarations in `navatala/_registry.hpp` and links
// against the trampoline in `_registry.cpp`. Partitioning keeps each
// translation unit well under the 50K-line threshold so a typical
// build can parallelise across the buckets.
#include "navatala/_registry.hpp"
#include <cstdint>
#include <string>
#include <string_view>

const char* k_cuda_navatala_samples_scatter_add_float32 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_samples_scatter_add_float32(const unsigned int* dstIdx, const float* messages, float* nodeFeatures) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  float scatterResult = atomicAdd(&(nodeFeatures[dstIdx[(int)(blockIdx.x * blockDim.x + threadIdx.x)]]), messages[(int)(blockIdx.x * blockDim.x + threadIdx.x)]);
}

)kernel";
const char* k_cuda_navatala_samples_float32_add = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_samples_float32_add(const float* a, const float* b, float* c) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  c[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (a[(int)(blockIdx.x * blockDim.x + threadIdx.x)] + b[(int)(blockIdx.x * blockDim.x + threadIdx.x)]);
}

)kernel";
const char* k_cuda_navatala_samples_triangle_normals2 = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_samples_triangle_normals2(const float* vertices, const int* faces, float* normals) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int i0 = faces[(((int)(blockIdx.x * blockDim.x + threadIdx.x))*3 + 0)];
  int i1 = faces[(((int)(blockIdx.x * blockDim.x + threadIdx.x))*3 + 1)];
  int i2 = faces[(((int)(blockIdx.x * blockDim.x + threadIdx.x))*3 + 2)];
  float p0x = vertices[((i0)*3 + 0)];
  float p0y = vertices[((i0)*3 + 1)];
  float p0z = vertices[((i0)*3 + 2)];
  float p1x = vertices[((i1)*3 + 0)];
  float p1y = vertices[((i1)*3 + 1)];
  float p1z = vertices[((i1)*3 + 2)];
  float p2x = vertices[((i2)*3 + 0)];
  float p2y = vertices[((i2)*3 + 1)];
  float p2z = vertices[((i2)*3 + 2)];
  float e1x = (p1x - p0x);
  float e1y = (p1y - p0y);
  float e1z = (p1z - p0z);
  float e2x = (p2x - p0x);
  float e2y = (p2y - p0y);
  float e2z = (p2z - p0z);
  float nx = ((e1y * e2z) - (e1z * e2y));
  float ny = ((e1z * e2x) - (e1x * e2z));
  float nz = ((e1x * e2y) - (e1y * e2x));
  normals[(((int)(blockIdx.x * blockDim.x + threadIdx.x))*3 + 0)] = nx;
  normals[(((int)(blockIdx.x * blockDim.x + threadIdx.x))*3 + 1)] = ny;
  normals[(((int)(blockIdx.x * blockDim.x + threadIdx.x))*3 + 2)] = nz;
}

)kernel";
const char* k_cuda_navatala_samples_axpy_fallback = R"kernel(
#include <cuda_runtime.h>
extern "C" __global__ void navatala_samples_axpy_fallback(const float* x, const float* y, const float* alpha, float* _out) {
  int gid0 = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  _out[(int)(blockIdx.x * blockDim.x + threadIdx.x)] = (y[(int)(blockIdx.x * blockDim.x + threadIdx.x)] + (x[(int)(blockIdx.x * blockDim.x + threadIdx.x)] * alpha[0]));
}

)kernel";

namespace NavatalaRegistry {

const KernelArgumentInfo kAbiArgs_cuda_navatala_samples_scatter_add_float32[] = {
  { "dstIdx", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "messages", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "nodeFeatures", 2, KernelArgumentRole::InputOutput, KernelAccessMode::ReadWrite, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_samples_scatter_add_float32 = {
  1,
  "navatala_samples_scatter_add_float32",
  "cuda",
  "navatala_samples_scatter_add_float32",
  "kernel:cuda:navatala_samples_scatter_add_float32",
  "abi-r1:cuda:navatala_samples_scatter_add_float32",
  "abi-r1:cuda:navatala_samples_scatter_add_float32",
  3,
  kAbiArgs_cuda_navatala_samples_scatter_add_float32
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_samples_float32_add[] = {
  { "a", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "b", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "c", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_samples_float32_add = {
  1,
  "navatala_samples_float32_add",
  "cuda",
  "navatala_samples_float32_add",
  "kernel:cuda:navatala_samples_float32_add",
  "abi-r1:cuda:navatala_samples_float32_add",
  "abi-r1:cuda:navatala_samples_float32_add",
  3,
  kAbiArgs_cuda_navatala_samples_float32_add
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_samples_triangle_normals2[] = {
  { "vertices", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "faces", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "normals", 2, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_samples_triangle_normals2 = {
  1,
  "navatala_samples_triangle_normals2",
  "cuda",
  "navatala_samples_triangle_normals2",
  "kernel:cuda:navatala_samples_triangle_normals2",
  "abi-r1:cuda:navatala_samples_triangle_normals2",
  "abi-r1:cuda:navatala_samples_triangle_normals2",
  3,
  kAbiArgs_cuda_navatala_samples_triangle_normals2
};

const KernelArgumentInfo kAbiArgs_cuda_navatala_samples_axpy_fallback[] = {
  { "x", 0, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "y", 1, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 },
  { "alpha", 2, KernelArgumentRole::Input, KernelAccessMode::ReadOnly, GpuRuntime::MemoryKind::Device, true, 4, 4, 256, nullptr, 0, 0 },
  { "out", 3, KernelArgumentRole::Output, KernelAccessMode::WriteOnly, GpuRuntime::MemoryKind::Device, true, 0, 0, 256, nullptr, 0, 0 }
};
const KernelAbiManifestInfo kAbiManifest_cuda_navatala_samples_axpy_fallback = {
  1,
  "navatala_samples_axpy_fallback",
  "cuda",
  "navatala_samples_axpy_fallback",
  "kernel:cuda:navatala_samples_axpy_fallback",
  "abi-r1:cuda:navatala_samples_axpy_fallback",
  "abi-r1:cuda:navatala_samples_axpy_fallback",
  4,
  kAbiArgs_cuda_navatala_samples_axpy_fallback
};

bool tryGetKernelAbiManifest_cuda_samples(const std::string& backend, const std::string& kernelName, const KernelAbiManifestInfo*& out) {
  if (backend == "cuda" && kernelName == "navatala_samples_scatter_add_float32") {
    out = &kAbiManifest_cuda_navatala_samples_scatter_add_float32;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_samples_float32_add") {
    out = &kAbiManifest_cuda_navatala_samples_float32_add;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_samples_triangle_normals2") {
    out = &kAbiManifest_cuda_navatala_samples_triangle_normals2;
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_samples_axpy_fallback") {
    out = &kAbiManifest_cuda_navatala_samples_axpy_fallback;
    return true;
  }
  out = nullptr;
  return false;
}

bool tryGetKernelSource_cuda_samples(const std::string& backend, const std::string& kernelName, GpuRuntime::ProgramSource& out) {
  if (backend == "cuda" && kernelName == "navatala_samples_scatter_add_float32") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_samples_scatter_add_float32";
    std::string_view sv(k_cuda_navatala_samples_scatter_add_float32);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_samples_float32_add") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_samples_float32_add";
    std::string_view sv(k_cuda_navatala_samples_float32_add);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_samples_triangle_normals2") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_samples_triangle_normals2";
    std::string_view sv(k_cuda_navatala_samples_triangle_normals2);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  if (backend == "cuda" && kernelName == "navatala_samples_axpy_fallback") {
    out.kind = GpuRuntime::ProgramSource::Kind::CudaCpp;
    out.entryPoint = "navatala_samples_axpy_fallback";
    std::string_view sv(k_cuda_navatala_samples_axpy_fallback);
    out.bytes.assign(sv.begin(), sv.end());
    return true;
  }
  return false;
}

} // namespace NavatalaRegistry

