// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Navatala Systems (OPC) Pvt Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Navatala C++ Wrapper Layer - C FFI Interface
// Part of the Navatala GPU Runtime Library
#ifndef NAVATALA_FFI_H
#define NAVATALA_FFI_H

#include <stddef.h>
#include <stdint.h>

/**
 * @file navatala/navatala_ffi.h
 * @brief C-compatible Foreign Function Interface for the Navatala GPU runtime.
 *
 * This header defines the stable C ABI that the C++ wrapper layer calls into.
 * The FFI provides opaque handle types and C functions for GPU resource
 * management, following the pattern of other GPU runtime libraries.
 *
 * All functions in this header are extern "C" and use C-compatible types only.
 */

#ifdef __cplusplus
extern "C" {
#define NAVATALA_NOEXCEPT noexcept
#else
#define NAVATALA_NOEXCEPT
#endif

#ifndef NAVATALA_GPU_FFI_ABI_VERSION
// Per #1513 §2.5: bumped from 5 → 6 alongside the kernel symbol/filename
// rename. Package metadata must expose the same ABI revision.
#define NAVATALA_GPU_FFI_ABI_VERSION 6
#endif

// ============================================================================
// Backend Enumeration (must match navatala/backend.hpp)
// ============================================================================

typedef enum NavatalaBackend {
    NAVATALA_BACKEND_CUDA_FFI   = 0,
    NAVATALA_BACKEND_HIP_FFI    = 1,
    NAVATALA_BACKEND_METAL_FFI  = 2,
    NAVATALA_BACKEND_OPENCL_FFI = 3,
    NAVATALA_BACKEND_VULKAN_FFI = 4,
    NAVATALA_BACKEND_AUTO_FFI   = 255
} NavatalaBackend;

// ============================================================================
// Error Codes (must match navatala/error.hpp)
// ============================================================================

typedef enum NavatalaErrorCode {
    NAVATALA_SUCCESS           = 0,
    NAVATALA_INVALID_HANDLE    = 1,
    NAVATALA_INVALID_PARAM     = 2,
    NAVATALA_GPU_ERROR         = 3,
    NAVATALA_OUT_OF_MEMORY     = 4,
    NAVATALA_NOT_IMPLEMENTED   = 5,
    NAVATALA_IO_ERROR          = 6,
    NAVATALA_RUNTIME_ERROR     = 7,
    NAVATALA_OVERFLOW_ERROR    = 8,
    NAVATALA_TIMEOUT           = 9,
    NAVATALA_CANCELLED         = 10,
    NAVATALA_NOT_FOUND         = 11,
    NAVATALA_ALREADY_EXISTS    = 12,
    NAVATALA_PERMISSION_DENIED = 13,
    NAVATALA_INTERNAL_ERROR    = 99
} NavatalaErrorCode;

// ============================================================================
// Opaque Handle Types
// ============================================================================

/** Opaque GPU context handle. */
typedef struct NavatalaGpuContext NavatalaGpuContext;

/** Opaque GPU queue/stream handle. */
typedef struct NavatalaGpuQueue NavatalaGpuQueue;

/** Opaque GPU event handle. */
typedef struct NavatalaGpuEvent NavatalaGpuEvent;

/** Opaque GPU buffer handle. */
typedef struct NavatalaGpuBuffer NavatalaGpuBuffer;

/** Opaque compiled GPU program handle. */
typedef struct NavatalaGpuProgram NavatalaGpuProgram;

/** Opaque memory resource handle. */
typedef struct NavatalaGpuMemoryResource NavatalaGpuMemoryResource;

/** Source or binary format used when creating a GPU program. */
typedef enum NavatalaProgramSourceKind {
    NAVATALA_PROGRAM_SOURCE_CUDA_CPP = 0,
    NAVATALA_PROGRAM_SOURCE_HIP_CPP = 1,
    NAVATALA_PROGRAM_SOURCE_OPENCL_C = 2,
    NAVATALA_PROGRAM_SOURCE_OPENCL_BINARY = 3,
    NAVATALA_PROGRAM_SOURCE_GLSL = 4,
    NAVATALA_PROGRAM_SOURCE_SPIRV = 5,
    NAVATALA_PROGRAM_SOURCE_MSL = 6,
    NAVATALA_PROGRAM_SOURCE_PTX = 7,
    NAVATALA_PROGRAM_SOURCE_CUBIN = 8,
    NAVATALA_PROGRAM_SOURCE_HSACO = 9,
    NAVATALA_PROGRAM_SOURCE_METALLIB = 10
} NavatalaProgramSourceKind;

/** Matrix operand orientation for GEMM-like operations. */
typedef enum NavatalaMatrixTranspose {
    NAVATALA_MATRIX_OP_NONE = 0,
    NAVATALA_MATRIX_OP_TRANSPOSE = 1
} NavatalaMatrixTranspose;

// ============================================================================
// Backend Capability Structure
// ============================================================================

/** Structure containing backend/device capabilities. */
typedef struct NavatalaBackendCapabilities {
    NavatalaBackend type;
    char name[128];
    char device_name[128];
    char driver_version[64];

    int major_version;
    int minor_version;
    int patch_version;

    // Memory capabilities
    uint8_t supports_managed_memory;
    uint8_t supports_unified_memory;
    uint8_t supports_virtual_memory;
    uint8_t supports_memory_pools;

    // Precision capabilities
    uint8_t supports_fp16;
    uint8_t supports_bf16;
    uint8_t supports_tf32;
    uint8_t supports_fp64;
    uint8_t supports_int8;
    uint8_t supports_tensor_cores;

    // Async capabilities
    uint8_t supports_streams;
    uint8_t supports_events;
    uint8_t supports_graphs;
    uint8_t supports_cooperative_groups;

    // Memory limits
    size_t max_global_memory_bytes;
    size_t max_shared_memory_bytes;
    size_t max_constant_memory_bytes;
    size_t max_register_bytes;
    size_t max_allocation_bytes;

    // Compute limits
    uint32_t max_thread_block_size;
    uint32_t max_grid_dim_x;
    uint32_t max_grid_dim_y;
    uint32_t max_grid_dim_z;
    int max_thread_blocks;
    int multiprocessor_count;
    int threads_per_warp;
    int max_warps_per_multiprocessor;
} NavatalaBackendCapabilities;

// ============================================================================
// Float16/BFloat16 Conversion Functions
// ============================================================================

/**
 * Convert float32 to IEEE 754 half-precision (float16).
 * @param f The float32 value to convert.
 * @return The float16 value as a uint16_t bit pattern.
 */
uint16_t navatala_float_to_half(float f) NAVATALA_NOEXCEPT;

/**
 * Convert IEEE 754 half-precision (float16) to float32.
 * @param h The float16 bit pattern.
 * @return The float32 value.
 */
float navatala_half_to_float(uint16_t h) NAVATALA_NOEXCEPT;

/**
 * Convert float32 to bfloat16.
 * @param f The float32 value to convert.
 * @return The bfloat16 value as a uint16_t bit pattern.
 */
uint16_t navatala_float_to_bfloat16(float f) NAVATALA_NOEXCEPT;

/**
 * Convert bfloat16 to float32.
 * @param b The bfloat16 bit pattern.
 * @return The float32 value.
 */
float navatala_bfloat16_to_float(uint16_t b) NAVATALA_NOEXCEPT;

// ============================================================================
// Backend Query Functions
// ============================================================================

/**
 * Get the number of available backends.
 * @return The number of available backends.
 */
int navatala_get_available_backend_count(void);

/**
 * Get available backends.
 * @param backends Output array to fill with available backends.
 * @param max_count Maximum number of backends to return.
 * @return The number of backends written to the array.
 */
int navatala_get_available_backends(NavatalaBackend* backends, int max_count);

/**
 * Check if a backend is available.
 * @param backend The backend to check.
 * @return 1 if available, 0 if not.
 */
int navatala_is_backend_available(NavatalaBackend backend);

/**
 * Get the capabilities of a backend.
 * @param backend The backend to query.
 * @param device_id The device ID (0-based).
 * @param caps Output structure to fill with capabilities.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_get_backend_capabilities(
    NavatalaBackend backend,
    int device_id,
    NavatalaBackendCapabilities* caps);

/**
 * Query current backend memory telemetry using backend runtime APIs.
 *
 * This reports live free/total memory only when the selected backend exposes a
 * reliable programmatic query. Backends without such support return
 * NAVATALA_SUCCESS with supported set to 0 and byte counts set to 0.
 *
 * @param backend The backend to query.
 * @param device_id The device ID (0-based).
 * @param free_bytes Output current free memory in bytes when supported.
 * @param total_bytes Output total memory in bytes when supported.
 * @param supported Output flag: 1 when byte counts are valid, 0 otherwise.
 * @return NAVATALA_SUCCESS on successful query or unsupported-clean result.
 */
NavatalaErrorCode navatala_get_backend_memory_info(
    NavatalaBackend backend,
    int device_id,
    size_t* free_bytes,
    size_t* total_bytes,
    uint8_t* supported);

/**
 * Get the number of devices for a backend.
 * @param backend The backend to query.
 * @return The number of devices, or 0 if the backend is unavailable.
 */
int navatala_get_device_count(NavatalaBackend backend);

/**
 * Get the current/default backend.
 * @return The currently selected backend.
 */
NavatalaBackend navatala_get_current_backend(void);

// ============================================================================
// Context Management
// ============================================================================

/**
 * Create a GPU context for the specified backend and device.
 * @param backend The backend to use (or AUTO for auto-selection).
 * @param device_id The device ID (0-based).
 * @param ctx Output pointer to receive the created context.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_create_context(
    NavatalaBackend backend,
    int device_id,
    NavatalaGpuContext** ctx);

/**
 * Destroy a GPU context.
 * @param ctx The context to destroy.
 */
void navatala_gpu_destroy_context(NavatalaGpuContext* ctx);

/**
 * Get the backend type of a context.
 * @param ctx The context to query.
 * @return The backend type.
 */
NavatalaBackend navatala_gpu_context_get_backend(NavatalaGpuContext* ctx);

/**
 * Get the device ID of a context.
 * @param ctx The context to query.
 * @return The device ID.
 */
int navatala_gpu_context_get_device_id(NavatalaGpuContext* ctx);

// ============================================================================
// Queue Management
// ============================================================================

/**
 * Create a GPU queue/stream.
 * @param ctx The context to create the queue in.
 * @param priority Queue priority (-1 = default, 0 = high).
 * @param queue Output pointer to receive the created queue.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_create_queue(
    NavatalaGpuContext* ctx,
    int priority,
    NavatalaGpuQueue** queue);

/**
 * Destroy a GPU queue.
 * @param queue The queue to destroy.
 */
void navatala_gpu_destroy_queue(NavatalaGpuQueue* queue);

/**
 * Synchronize a queue (block until all operations complete).
 * @param queue The queue to synchronize.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_queue_sync(NavatalaGpuQueue* queue);

/**
 * Return a backend-native queue/stream handle as an opaque integer token.
 *
 * The token is intended for interop protocols such as DLPack where Python
 * frameworks exchange stream identifiers. A return status of NAVATALA_SUCCESS
 * with `is_supported == 0` means the queue is valid but this backend/build
 * does not expose a DLPack-compatible native stream token.
 *
 * DLPack sentinel values 0, 1, and 2 are reserved and are not produced by this
 * function for ordinary native handles. Callers receiving Python stream values
 * must still treat those sentinel values specially and never cast them to
 * pointers.
 */
NavatalaErrorCode navatala_gpu_queue_native_handle(
    NavatalaGpuQueue* queue,
    uintptr_t* stream_handle,
    uint8_t* is_supported);

/**
 * Check if a queue is ready (all operations completed).
 * @param queue The queue to check.
 * @return 1 if ready, 0 if still executing.
 */
int navatala_gpu_queue_is_ready(NavatalaGpuQueue* queue);

/**
 * Get the backend type of a queue.
 * @param queue The queue to query.
 * @return The backend type.
 */
NavatalaBackend navatala_gpu_queue_get_backend(NavatalaGpuQueue* queue);

// ============================================================================
// Event Management
// ============================================================================

/**
 * Create a GPU event.
 * @param ctx The context to create the event in.
 * @param event Output pointer to receive the created event.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_create_event(
    NavatalaGpuContext* ctx,
    NavatalaGpuEvent** event);

/**
 * Destroy a GPU event.
 * @param event The event to destroy.
 */
void navatala_gpu_destroy_event(NavatalaGpuEvent* event);

/**
 * Record an event in a queue.
 * @param event The event to record.
 * @param queue The queue to record in.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_event_record(
    NavatalaGpuEvent* event,
    NavatalaGpuQueue* queue);

/**
 * Make a queue wait for an event.
 * @param queue The queue that will wait.
 * @param event The event to wait for.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_event_wait(
    NavatalaGpuQueue* queue,
    NavatalaGpuEvent* event);

/**
 * Check if an event has completed.
 * @param event The event to check.
 * @return 1 if complete, 0 if pending.
 */
int navatala_gpu_event_is_complete(NavatalaGpuEvent* event);

/**
 * Block the host thread until an event completes.
 * This is more efficient than polling is_complete() in a loop.
 * @param event The event to synchronize with.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_event_synchronize(NavatalaGpuEvent* event);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Allocate device memory.
 * @param ctx The context to allocate in.
 * @param bytes Number of bytes to allocate.
 * @param alignment Required alignment (typically 256).
 * @param ptr Output pointer to receive the allocated memory.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_allocate(
    NavatalaGpuContext* ctx,
    size_t bytes,
    size_t alignment,
    void** ptr);

/**
 * Deallocate device memory.
 * @param ctx The context the memory was allocated in.
 * @param ptr The pointer to deallocate.
 * @param bytes The size of the allocation.
 */
void navatala_gpu_deallocate(
    NavatalaGpuContext* ctx,
    void* ptr,
    size_t bytes);

// ============================================================================
// Buffer Operations
// ============================================================================

/**
 * Create a device buffer.
 * @param ctx The context to create the buffer in.
 * @param bytes Number of bytes for the buffer.
 * @param buffer Output pointer to receive the created buffer.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_create_buffer(
    NavatalaGpuContext* ctx,
    size_t bytes,
    NavatalaGpuBuffer** buffer);

/**
 * Destroy a device buffer.
 * @param buffer The buffer to destroy.
 */
void navatala_gpu_destroy_buffer(NavatalaGpuBuffer* buffer);

/**
 * Get the size of a buffer in bytes.
 * @param buffer The buffer to query.
 * @return The buffer size in bytes.
 */
size_t navatala_gpu_buffer_size(NavatalaGpuBuffer* buffer);

/**
 * Get the device pointer of a buffer.
 * @param buffer The buffer to query.
 * @return The device pointer.
 */
void* navatala_gpu_buffer_ptr(NavatalaGpuBuffer* buffer);

/**
 * Copy data from host to device.
 * @param buffer The destination buffer.
 * @param host_ptr The source host pointer.
 * @param bytes Number of bytes to copy.
 * @param queue The queue to use for the copy (or NULL for default).
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_copy_h2d(
    NavatalaGpuBuffer* buffer,
    const void* host_ptr,
    size_t bytes,
    NavatalaGpuQueue* queue);

/**
 * Copy data from device to host.
 * @param buffer The source buffer.
 * @param host_ptr The destination host pointer.
 * @param bytes Number of bytes to copy.
 * @param queue The queue to use for the copy (or NULL for default).
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_copy_d2h(
    const NavatalaGpuBuffer* buffer,
    void* host_ptr,
    size_t bytes,
    NavatalaGpuQueue* queue);

/**
 * Copy data between device buffers.
 * @param dst The destination buffer.
 * @param src The source buffer.
 * @param bytes Number of bytes to copy.
 * @param queue The queue to use for the copy (or NULL for default).
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_copy_d2d(
    NavatalaGpuBuffer* dst,
    const NavatalaGpuBuffer* src,
    size_t bytes,
    NavatalaGpuQueue* queue);

// ============================================================================
// Portable Operation Helpers
// ============================================================================

/**
 * Compute y = alpha * x + y for single-precision vectors.
 *
 * This ABI entrypoint is intentionally stable and backend-neutral. Runtime
 * implementations may dispatch to a tuned device kernel/library path; they must
 * preserve the same validation and result semantics.
 *
 * @param x Input vector buffer containing at least element_count float values.
 * @param y Input/output vector buffer containing at least element_count floats.
 * @param alpha Scalar multiplier.
 * @param element_count Number of float32 elements to process.
 * @param queue Optional queue for associated device copies/dispatch.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_axpy_f32(
    const NavatalaGpuBuffer* x,
    NavatalaGpuBuffer* y,
    float alpha,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute C = alpha * A @ B + beta * C for row-major single-precision matrices.
 *
 * A has shape [m, k], B has shape [k, n], and C has shape [m, n].
 * Runtime implementations may dispatch to backend BLAS or kernel-runtime
 * launchers; this ABI preserves a uniform validation/result contract for Python and C++
 * wrappers.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_gemm_f32(
    const NavatalaGpuBuffer* a,
    const NavatalaGpuBuffer* b,
    NavatalaGpuBuffer* c,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    float beta,
    NavatalaGpuQueue* queue);

/**
 * Compute C = A @ B for row-major half-precision input matrices with
 * single-precision accumulation/output.
 *
 * A has shape [m, k], B has shape [k, n], and C has shape [m, n]. This wrapper
 * is the public entry point for generated HIP/gfx942 MFMA kernels. The current
 * production MFMA path supports NN layout with F32 alpha/beta epilogue scaling
 * and edge-tile bounds checks. Transpose and batched modes are exposed through
 * navatala_gpu_gemm_f16_f32_ex and navatala_gpu_gemm_f16_f32_strided_batched.
 * Unsupported forced-MFMA calls return a loud error instead of silently
 * launching the wrong kernel.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_gemm_f16_f32(
    const NavatalaGpuBuffer* a,
    const NavatalaGpuBuffer* b,
    NavatalaGpuBuffer* c,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    float beta,
    NavatalaGpuQueue* queue);

/**
 * Compute C = alpha * op(A) @ op(B) + beta * C for row-major half-precision
 * input matrices with single-precision accumulation/output.
 *
 * Logical op(A) has shape [m, k], logical op(B) has shape [k, n], and C has
 * shape [m, n]. When trans_a is NONE, A is stored as [m, k]; when TRANSPOSE,
 * A is stored as [k, m]. B follows the analogous [k, n] / [n, k] rule.
 *
 * The current HIP/gfx942 MFMA fast path supports all four orientation
 * combinations for single-batch calls and delegates strided-batched calls to
 * navatala_gpu_gemm_f16_f32_strided_batched.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_gemm_f16_f32_ex(
    const NavatalaGpuBuffer* a,
    const NavatalaGpuBuffer* b,
    NavatalaGpuBuffer* c,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    float beta,
    NavatalaMatrixTranspose trans_a,
    NavatalaMatrixTranspose trans_b,
    NavatalaGpuQueue* queue);

/**
 * Strided-batched form of navatala_gpu_gemm_f16_f32_ex.
 *
 * Strides are expressed in elements, not bytes. Each batch computes:
 *   C_i = alpha * op(A_i) @ op(B_i) + beta * C_i
 * where A_i/B_i/C_i start at base + i * stride_{a,b,c}. Strides are expressed
 * in elements, not bytes. A zero stride for A and/or B broadcasts that input
 * matrix across all batches; C must use a non-overlapping positive stride for
 * multi-batch calls to avoid concurrent writes to the same output matrix. On
 * HIP/gfx942 the MFMA fast path launches one 3D grid across all batches; other
 * backends use the portable correctness fallback until backend-specific batched
 * kernels are introduced.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_gemm_f16_f32_strided_batched(
    const NavatalaGpuBuffer* a,
    const NavatalaGpuBuffer* b,
    NavatalaGpuBuffer* c,
    size_t m,
    size_t n,
    size_t k,
    float alpha,
    float beta,
    NavatalaMatrixTranspose trans_a,
    NavatalaMatrixTranspose trans_b,
    size_t stride_a,
    size_t stride_b,
    size_t stride_c,
    size_t batch_count,
    NavatalaGpuQueue* queue);

/**
 * Compute result[0] = ||x||_2 for a single-precision vector.
 *
 * The result buffer must contain space for one float32 value. This ABI helper
 * uses the same backend-neutral tensor-buffer contract as AXPY/GEMM so Python
 * bindings never need to pass raw device pointers.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_nrm2_f32(
    const NavatalaGpuBuffer* x,
    NavatalaGpuBuffer* result,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute a null-aware Float32 column sum.
 *
 * input has length element_count. input_valid is a uint32 validity bitmap with
 * ceil(element_count / 32) words. output is a single float32 scalar buffer and
 * output_valid is a uint32 bitmap whose bit 0 is set when at least one input
 * element is valid. Null elements are skipped.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_reduce_sum_f32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute a null-aware Int32 column sum widened to Int64.
 *
 * input has length element_count. input_valid is a uint32 validity bitmap with
 * ceil(element_count / 32) words. output is a single int64 scalar buffer and
 * output_valid is a uint32 bitmap whose bit 0 is set when at least one input
 * element is valid. Null elements are skipped and valid int32 values are
 * widened before accumulation.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_reduce_sum_i32_to_i64(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute a null-aware Int32 column minimum.
 *
 * input has length element_count. input_valid is a uint32 validity bitmap with
 * ceil(element_count / 32) words. output is a single int32 scalar buffer and
 * output_valid is a uint32 bitmap whose bit 0 is set when at least one input
 * element is valid. Null elements are skipped.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_reduce_min_i32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute a null-aware Int32 column maximum.
 *
 * input has length element_count. input_valid is a uint32 validity bitmap with
 * ceil(element_count / 32) words. output is a single int32 scalar buffer and
 * output_valid is a uint32 bitmap whose bit 0 is set when at least one input
 * element is valid. Null elements are skipped.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_reduce_max_i32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Replace null float32 column entries with a scalar fill value.
 *
 * input and output have length element_count. input_valid and output_valid are
 * uint32 validity bitmaps with ceil(element_count / 32) words. The output
 * validity bitmap is overwritten so all rows in [0, element_count) are valid
 * and unused trailing bits in the final word are zero.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_fill_null_f32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    float fill_value,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Replace null int32 column entries with a scalar fill value.
 *
 * input and output have length element_count. input_valid and output_valid are
 * uint32 validity bitmaps with ceil(element_count / 32) words. The output
 * validity bitmap is overwritten so all rows in [0, element_count) are valid
 * and unused trailing bits in the final word are zero.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_fill_null_i32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    int32_t fill_value,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Gather float32 DataFrame values by int32 indices with validity propagation.
 *
 * output[i] = input[indices[i]] when indices[i] is in range and the source row
 * is valid. Out-of-range and null source rows produce output value 0 and clear
 * the corresponding output validity bit. output_valid is overwritten with
 * ceil(output_count / 32) uint32 words.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_gather_f32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    const NavatalaGpuBuffer* indices,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t input_count,
    size_t output_count,
    NavatalaGpuQueue* queue);

/**
 * Gather int32 DataFrame values by int32 indices with validity propagation.
 *
 * output[i] = input[indices[i]] when indices[i] is in range and the source row
 * is valid. Out-of-range and null source rows produce output value 0 and clear
 * the corresponding output validity bit. output_valid is overwritten with
 * ceil(output_count / 32) uint32 words.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_gather_i32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    const NavatalaGpuBuffer* indices,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t input_count,
    size_t output_count,
    NavatalaGpuQueue* queue);

/**
 * Scatter float32 DataFrame values by int32 destination indices with validity propagation.
 *
 * output[indices[i]] = input[i] when indices[i] is in range. Valid source rows
 * set the corresponding output validity bit; invalid source rows still scatter
 * their value but do not set validity. Duplicate destination indices have
 * contract-level undefined ordering, matching the GPU kernel.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_scatter_f32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    const NavatalaGpuBuffer* indices,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t input_count,
    size_t output_count,
    NavatalaGpuQueue* queue);

/**
 * Scatter int32 DataFrame values by int32 destination indices with validity propagation.
 *
 * output[indices[i]] = input[i] when indices[i] is in range. Valid source rows
 * set the corresponding output validity bit; invalid source rows still scatter
 * their value but do not set validity. Duplicate destination indices have
 * contract-level undefined ordering, matching the GPU kernel.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_scatter_i32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    const NavatalaGpuBuffer* indices,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t input_count,
    size_t output_count,
    NavatalaGpuQueue* queue);

/**
 * Compact float32 DataFrame values by a uint32 selection mask.
 *
 * For every row where mask[i] == 1, output[prefix_sum[i]] receives input[i].
 * Source validity is propagated to output_valid. prefix_sum must be the
 * exclusive prefix sum of the mask and all selected positions must be in range
 * for the supplied output_count.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_compact_by_mask_f32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    const NavatalaGpuBuffer* mask,
    const NavatalaGpuBuffer* prefix_sum,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t input_count,
    size_t output_count,
    NavatalaGpuQueue* queue);

/**
 * Compact int32 DataFrame values by a uint32 selection mask.
 *
 * For every row where mask[i] == 1, output[prefix_sum[i]] receives input[i].
 * Source validity is propagated to output_valid. prefix_sum must be the
 * exclusive prefix sum of the mask and all selected positions must be in range
 * for the supplied output_count.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_compact_by_mask_i32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    const NavatalaGpuBuffer* mask,
    const NavatalaGpuBuffer* prefix_sum,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t input_count,
    size_t output_count,
    NavatalaGpuQueue* queue);

/**
 * Coalesce two nullable float32 DataFrame columns.
 *
 * output[i] = col1[i] when col1_valid[i] is set, otherwise col2[i] when
 * col2_valid[i] is set. Rows where neither input is valid produce output value
 * 0 and clear the corresponding output validity bit. output_valid is
 * overwritten with ceil(element_count / 32) uint32 words.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_coalesce_f32(
    const NavatalaGpuBuffer* col1,
    const NavatalaGpuBuffer* col1_valid,
    const NavatalaGpuBuffer* col2,
    const NavatalaGpuBuffer* col2_valid,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Copy float32 values where source and destination rows are both valid.
 *
 * dst is read/write: rows whose src_valid and dst_valid bits are both set are
 * replaced with src values, and all other dst rows are left unchanged.
 * out_valid is overwritten with the intersection of src_valid and dst_valid for
 * rows in [0, element_count); unused trailing bits in the final word are zero.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_copy_if_valid_f32(
    const NavatalaGpuBuffer* src,
    const NavatalaGpuBuffer* src_valid,
    const NavatalaGpuBuffer* dst_valid,
    NavatalaGpuBuffer* dst,
    NavatalaGpuBuffer* out_valid,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Inclusive prefix sum for a nullable float32 DataFrame column.
 *
 * Null input rows contribute 0 to the prefix sum. output_valid[i] is set only
 * when every input row in the prefix [0, i] is valid, matching the formal
 * single-workgroup DataFrame scan contract. element_count must be <= 256.
 * output_valid is overwritten with ceil(element_count / 32) uint32 words.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_inclusive_scan_sum_f32(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Inclusive prefix sum for a nullable int32 DataFrame column, widened to int64.
 *
 * Null input rows contribute 0 to the running sum. output_valid follows
 * input_valid: null rows are null in the output, but later valid rows include
 * the accumulated sum across null rows. element_count must be <= 256.
 * output_valid is overwritten with ceil(element_count / 32) uint32 words.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_inclusive_scan_sum_i32_to_i64_skip_nulls(
    const NavatalaGpuBuffer* input,
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    NavatalaGpuBuffer* output_valid,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Count valid entries in a uint32 validity bitmap.
 *
 * input_valid has ceil(element_count / 32) words. output is a scalar uint32
 * buffer. Unused trailing bits in the final bitmap word are ignored.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_count_valid_u32(
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Expand a uint32 validity bitmap to a uint8 mask where 1 means valid.
 *
 * input_valid has ceil(element_count / 32) words. output has element_count
 * uint8 entries.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_is_valid_u8(
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Expand a uint32 validity bitmap to a uint8 mask where 1 means null.
 *
 * input_valid has ceil(element_count / 32) words. output has element_count
 * uint8 entries.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_is_null_u8(
    const NavatalaGpuBuffer* input_valid,
    NavatalaGpuBuffer* output,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Fill a float32 output column with a scalar value.
 *
 * output has element_count float32 entries. element_count must fit in uint32.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_fill_constant_f32(
    NavatalaGpuBuffer* output,
    float value,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Fill an int32 output column with a scalar value.
 *
 * output has element_count int32 entries. element_count must fit in uint32.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_fill_constant_i32(
    NavatalaGpuBuffer* output,
    int32_t value,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Fill an int32 DataFrame column with a sequential range.
 *
 * output[i] = start + i for i in [0, element_count). The runtime rejects
 * ranges that cannot be represented as int32 values.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_fill_range_i32(
    NavatalaGpuBuffer* output,
    int32_t start,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Fill a uint32 validity bitmap so all rows are valid.
 *
 * output has ceil(element_count / 32) uint32 words. Unused trailing bits in the
 * final word are zero.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_fill_validity_all_valid_u32(
    NavatalaGpuBuffer* output,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Fill a uint32 validity bitmap so all rows are null.
 *
 * output has ceil(element_count / 32) uint32 words.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_dataframe_fill_validity_all_null_u32(
    NavatalaGpuBuffer* output,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute result[0] = mean((y_true - y_pred)^2) for float32 vectors.
 *
 * y_true and y_pred each have length element_count. result is a scalar float32
 * buffer. element_count must be non-zero because MSE is undefined for empty
 * vectors.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_ml_mse_f32(
    const NavatalaGpuBuffer* y_true,
    const NavatalaGpuBuffer* y_pred,
    NavatalaGpuBuffer* result,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute the single-precision root mean squared error over two vectors.
 *
 * y_true and y_pred each have length element_count. result is a scalar float32
 * buffer. element_count must be non-zero because RMSE is undefined for empty
 * vectors.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_ml_rmse_f32(
    const NavatalaGpuBuffer* y_true,
    const NavatalaGpuBuffer* y_pred,
    NavatalaGpuBuffer* result,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute the single-precision mean absolute error over two vectors.
 *
 * y_true and y_pred must each contain element_count float32 values. result is a
 * scalar float32 buffer. element_count must be non-zero and fit in uint32.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_ml_mae_f32(
    const NavatalaGpuBuffer* y_true,
    const NavatalaGpuBuffer* y_pred,
    NavatalaGpuBuffer* result,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute single-precision mean absolute percentage error over two vectors.
 *
 * y_true and y_pred must each contain element_count float32 values. result is a
 * scalar float32 buffer. element_count must be non-zero and fit in uint32.
 * Terms with y_true == 0 follow IEEE-754 division semantics, matching the
 * formal stdlib contract.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_ml_mape_f32(
    const NavatalaGpuBuffer* y_true,
    const NavatalaGpuBuffer* y_pred,
    NavatalaGpuBuffer* result,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute single-precision R2 score over two vectors.
 *
 * y_true and y_pred must each contain element_count float32 values. y_mean and
 * result are scalar float32 buffers. element_count must be non-zero and fit in
 * uint32. Constant y_true inputs follow IEEE-754 division semantics, matching
 * the formal stdlib contract.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_ml_r2_score_f32(
    const NavatalaGpuBuffer* y_true,
    const NavatalaGpuBuffer* y_pred,
    const NavatalaGpuBuffer* y_mean,
    NavatalaGpuBuffer* result,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute single-precision explained variance score over two vectors.
 *
 * y_true and y_pred must each contain element_count float32 values.
 * mean_residuals, mean_true, and result are scalar float32 buffers.
 * element_count must be non-zero and fit in uint32. Constant y_true inputs
 * follow IEEE-754 division semantics, matching the formal stdlib contract.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_ml_explained_variance_score_f32(
    const NavatalaGpuBuffer* y_true,
    const NavatalaGpuBuffer* y_pred,
    const NavatalaGpuBuffer* mean_residuals,
    const NavatalaGpuBuffer* mean_true,
    NavatalaGpuBuffer* result,
    size_t element_count,
    NavatalaGpuQueue* queue);

/**
 * Compute y = A * x for a single-precision CSR sparse matrix.
 *
 * The CSR descriptor uses uint32 row offsets and column indices:
 * - row_offsets has length rows + 1 and must start at 0 and end at nnz
 * - col_indices has length nnz and each entry must be less than cols
 * - values has length nnz
 * - x has length cols
 * - y has length rows
 *
 * Runtime implementations may dispatch to backend sparse libraries or
 * kernel-runtime launchers; this ABI preserves a stable buffer-level contract for
 * Python/C++ wrappers.
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_csr_spmv_f32(
    const NavatalaGpuBuffer* row_offsets,
    const NavatalaGpuBuffer* col_indices,
    const NavatalaGpuBuffer* values,
    const NavatalaGpuBuffer* x,
    NavatalaGpuBuffer* y,
    size_t rows,
    size_t cols,
    size_t nnz,
    NavatalaGpuQueue* queue);

/**
 * Compute per-vertex out-degree from CSR row offsets.
 *
 * row_offsets has length vertex_count + 1 and degrees has length vertex_count.
 * Both buffers use uint32 elements. The implementation validates monotonic CSR
 * offsets and writes degrees[v] = row_offsets[v + 1] - row_offsets[v].
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_graph_out_degree_u32(
    const NavatalaGpuBuffer* row_offsets,
    NavatalaGpuBuffer* degrees,
    size_t vertex_count,
    NavatalaGpuQueue* queue);

/**
 * Compute per-vertex in-degree from CSC/transposed-CSR column offsets.
 *
 * col_offsets has length vertex_count + 1 and degrees has length vertex_count.
 * Both buffers use uint32 elements. The implementation validates monotonic CSC
 * offsets and writes degrees[v] = col_offsets[v + 1] - col_offsets[v].
 *
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_graph_in_degree_u32(
    const NavatalaGpuBuffer* col_offsets,
    NavatalaGpuBuffer* degrees,
    size_t vertex_count,
    NavatalaGpuQueue* queue);

// ============================================================================
// Program Creation And Raw Launch (Internal/Advanced)
// ============================================================================

/**
 * Create a backend program from source text or binary bytes.
 *
 * This is the low-level ABI used by the internal Python kernel-launch
 * helpers. Stable public Python APIs should normally prefer operation-level
 * wrappers such as navatala_gpu_axpy_f32/navatala_gpu_gemm_f32.
 *
 * @param ctx Context that owns the program.
 * @param kind Source/binary format.
 * @param source_bytes Source or binary payload.
 * @param source_size Payload size in bytes.
 * @param entry_point Kernel entry-point name.
 * @param program Output program handle.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_create_program(
    NavatalaGpuContext* ctx,
    NavatalaProgramSourceKind kind,
    const void* source_bytes,
    size_t source_size,
    const char* entry_point,
    NavatalaGpuProgram** program);

/**
 * Destroy a program created by navatala_gpu_create_program.
 * @param program The program to destroy.
 */
void navatala_gpu_destroy_program(NavatalaGpuProgram* program);

/**
 * Launch a program using positional buffer arguments.
 *
 * All buffers must belong to the same backend/device as the queue and program.
 * Scalar kernel arguments are intentionally not part of this v1 internal ABI;
 * stable public operation wrappers are used for scalar-rich APIs.
 *
 * @return NAVATALA_SUCCESS on successful launch/enqueue.
 */
NavatalaErrorCode navatala_gpu_launch_program(
    NavatalaGpuQueue* queue,
    NavatalaGpuProgram* program,
    NavatalaGpuBuffer* const* buffers,
    size_t buffer_count,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z,
    uint32_t block_x,
    uint32_t block_y,
    uint32_t block_z);

// ============================================================================
// Offset-Aware Copy Functions (P1: Dataset Slicing Support)
// ============================================================================

/**
 * Copy data from host to device at an offset within the buffer.
 * @param buffer The destination buffer.
 * @param buffer_offset_bytes Byte offset into the destination buffer.
 * @param host_ptr The source host pointer.
 * @param bytes Number of bytes to copy.
 * @param queue The queue to use for the copy (or NULL for default).
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_copy_h2d_offset(
    NavatalaGpuBuffer* buffer,
    size_t buffer_offset_bytes,
    const void* host_ptr,
    size_t bytes,
    NavatalaGpuQueue* queue);

/**
 * Copy data from device at an offset to host.
 * @param buffer The source buffer.
 * @param buffer_offset_bytes Byte offset into the source buffer.
 * @param host_ptr The destination host pointer.
 * @param bytes Number of bytes to copy.
 * @param queue The queue to use for the copy (or NULL for default).
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_copy_d2h_offset(
    const NavatalaGpuBuffer* buffer,
    size_t buffer_offset_bytes,
    void* host_ptr,
    size_t bytes,
    NavatalaGpuQueue* queue);

/**
 * Copy data between device buffers with offsets.
 * @param dst The destination buffer.
 * @param dst_offset_bytes Byte offset into the destination buffer.
 * @param src The source buffer.
 * @param src_offset_bytes Byte offset into the source buffer.
 * @param bytes Number of bytes to copy.
 * @param queue The queue to use for the copy (or NULL for default).
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_copy_d2d_offset(
    NavatalaGpuBuffer* dst,
    size_t dst_offset_bytes,
    const NavatalaGpuBuffer* src,
    size_t src_offset_bytes,
    size_t bytes,
    NavatalaGpuQueue* queue);

/**
 * Wrap an existing device pointer as a buffer (non-owning).
 * @param ctx The context.
 * @param ptr The device pointer to wrap.
 * @param bytes Size of the memory region.
 * @param buffer Output pointer to receive the wrapper buffer.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 * @warning The caller is responsible for the lifetime of the wrapped pointer.
 */
NavatalaErrorCode navatala_gpu_wrap_device_ptr(
    NavatalaGpuContext* ctx,
    void* ptr,
    size_t bytes,
    NavatalaGpuBuffer** buffer);

/**
 * Fill buffer with a pattern.
 * @param buffer The buffer to fill.
 * @param pattern The byte pattern to fill with.
 * @param queue The queue to use for the operation.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_buffer_fill(
    NavatalaGpuBuffer* buffer,
    uint8_t pattern,
    NavatalaGpuQueue* queue);

// ============================================================================
// Memory Resource (Pool) Management
// ============================================================================

/**
 * Create a memory pool resource.
 * @param ctx The context to create the pool in.
 * @param initial_size Initial pool size in bytes (0 for default).
 * @param max_size Maximum pool size in bytes (0 for unlimited).
 * @param resource Output pointer to receive the created resource.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_create_memory_pool(
    NavatalaGpuContext* ctx,
    size_t initial_size,
    size_t max_size,
    NavatalaGpuMemoryResource** resource);

/**
 * Destroy a memory resource.
 * @param resource The resource to destroy.
 */
void navatala_gpu_destroy_memory_resource(NavatalaGpuMemoryResource* resource);

/**
 * Allocate from a memory resource.
 * @param resource The resource to allocate from.
 * @param bytes Number of bytes to allocate.
 * @param alignment Required alignment.
 * @param ptr Output pointer to receive the allocated memory.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_gpu_resource_allocate(
    NavatalaGpuMemoryResource* resource,
    size_t bytes,
    size_t alignment,
    void** ptr);

/**
 * Deallocate from a memory resource.
 * @param resource The resource to deallocate from.
 * @param ptr The pointer to deallocate.
 * @param bytes The size of the allocation.
 */
void navatala_gpu_resource_deallocate(
    NavatalaGpuMemoryResource* resource,
    void* ptr,
    size_t bytes);

/**
 * Get statistics from a memory resource.
 * @param resource The resource to query.
 * @param allocated_bytes Output for current allocated bytes.
 * @param capacity_bytes Output for total capacity.
 * @param free_bytes Output for available bytes.
 */
void navatala_gpu_resource_stats(
    NavatalaGpuMemoryResource* resource,
    size_t* allocated_bytes,
    size_t* capacity_bytes,
    size_t* free_bytes);

/**
 * Check if a resource owns a pointer.
 * @param resource The resource to check.
 * @param ptr The pointer to check.
 * @return 1 if the resource owns the pointer, 0 otherwise.
 */
int navatala_gpu_resource_owns(
    NavatalaGpuMemoryResource* resource,
    void* ptr);

// ============================================================================
// Vector Search Types
// ============================================================================

/** Opaque vector search index handle. */
typedef struct NavatalaVectorIndex NavatalaVectorIndex;

/** CAGRA build parameters. */
typedef struct NavatalaCagraBuildParams {
    size_t graph_degree;
    size_t intermediate_graph_degree;
    int build_algo;
    size_t nn_descent_niter;
    int metric;
} NavatalaCagraBuildParams;

/** CAGRA search parameters. */
typedef struct NavatalaCagraSearchParams {
    size_t max_iterations;
    size_t itopk_size;
    size_t search_width;
    size_t min_iterations;
    size_t thread_block_size;
    int hashmap_mode;
    float hashmap_max_fill_rate;
    size_t num_random_samplings;
    uint64_t rand_xor_mask;
} NavatalaCagraSearchParams;

/** IVF-PQ build parameters. */
typedef struct NavatalaIvfpqBuildParams {
    size_t n_lists;
    size_t pq_dim;
    size_t pq_bits;
    size_t kmeans_n_iters;
    float kmeans_trainset_fraction;
    int metric;
} NavatalaIvfpqBuildParams;

/** IVF-PQ search parameters. */
typedef struct NavatalaIvfpqSearchParams {
    size_t n_probes;
    int lut_dtype;
    int internal_distance_dtype;
    float preferred_shmem_carveout;
} NavatalaIvfpqSearchParams;

/** IVF-Flat build parameters. */
typedef struct NavatalaIvfflatBuildParams {
    size_t n_lists;
    size_t kmeans_n_iters;
    float kmeans_trainset_fraction;
    int metric;
} NavatalaIvfflatBuildParams;

/** IVF-Flat search parameters. */
typedef struct NavatalaIvfflatSearchParams {
    size_t n_probes;
} NavatalaIvfflatSearchParams;

// ============================================================================
// Vector Index Management
// ============================================================================

/**
 * Destroy a vector index.
 * @param index The index to destroy.
 */
void navatala_vector_index_destroy(NavatalaVectorIndex* index);

// ============================================================================
// CAGRA Index Functions
// ============================================================================

/**
 * Build a CAGRA index.
 */
NavatalaErrorCode navatala_cagra_build(
    NavatalaGpuContext* ctx,
    const void* data,
    size_t n_vectors,
    size_t dims,
    const NavatalaCagraBuildParams* params,
    NavatalaVectorIndex** index);

/**
 * Load a CAGRA index from file.
 */
NavatalaErrorCode navatala_cagra_load(
    NavatalaGpuContext* ctx,
    const char* path,
    NavatalaVectorIndex** index,
    size_t* dims,
    size_t* n_vectors,
    size_t* graph_degree,
    int* metric);

/**
 * Search a CAGRA index.
 */
NavatalaErrorCode navatala_cagra_search(
    NavatalaVectorIndex* index,
    const void* queries,
    size_t n_queries,
    size_t k,
    const NavatalaCagraSearchParams* params,
    void* indices,
    void* distances,
    NavatalaGpuQueue* queue);

/**
 * Save a CAGRA index to file.
 */
NavatalaErrorCode navatala_cagra_save(
    NavatalaVectorIndex* index,
    const char* path);

// ============================================================================
// IVF-PQ Index Functions
// ============================================================================

NavatalaErrorCode navatala_ivfpq_build(
    NavatalaGpuContext* ctx,
    const void* data,
    size_t n_vectors,
    size_t dims,
    const NavatalaIvfpqBuildParams* params,
    NavatalaVectorIndex** index);

NavatalaErrorCode navatala_ivfpq_load(
    NavatalaGpuContext* ctx,
    const char* path,
    NavatalaVectorIndex** index,
    size_t* dims,
    size_t* n_vectors,
    size_t* n_lists,
    size_t* pq_dim,
    size_t* pq_bits,
    int* metric);

NavatalaErrorCode navatala_ivfpq_search(
    NavatalaVectorIndex* index,
    const void* queries,
    size_t n_queries,
    size_t k,
    const NavatalaIvfpqSearchParams* params,
    void* indices,
    void* distances,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_ivfpq_save(
    NavatalaVectorIndex* index,
    const char* path);

// ============================================================================
// IVF-Flat Index Functions
// ============================================================================

NavatalaErrorCode navatala_ivfflat_build(
    NavatalaGpuContext* ctx,
    const void* data,
    size_t n_vectors,
    size_t dims,
    const NavatalaIvfflatBuildParams* params,
    NavatalaVectorIndex** index);

NavatalaErrorCode navatala_ivfflat_load(
    NavatalaGpuContext* ctx,
    const char* path,
    NavatalaVectorIndex** index,
    size_t* dims,
    size_t* n_vectors,
    size_t* n_lists,
    int* metric);

NavatalaErrorCode navatala_ivfflat_search(
    NavatalaVectorIndex* index,
    const void* queries,
    size_t n_queries,
    size_t k,
    const NavatalaIvfflatSearchParams* params,
    void* indices,
    void* distances,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_ivfflat_save(
    NavatalaVectorIndex* index,
    const char* path);

// ============================================================================
// Brute Force Search
// ============================================================================

NavatalaErrorCode navatala_brute_force_search(
    NavatalaGpuContext* ctx,
    const void* database,
    size_t n_vectors,
    size_t dims,
    const void* queries,
    size_t n_queries,
    size_t k,
    int metric,
    void* indices,
    void* distances,
    NavatalaGpuQueue* queue);

// ============================================================================
// Distance Functions
// ============================================================================

NavatalaErrorCode navatala_pairwise_distance(
    const void* x, size_t m,
    const void* y, size_t n,
    size_t dims,
    int metric,
    void* output,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_fused_l2_nn(
    const void* x, size_t m,
    const void* y, size_t n,
    size_t dims,
    void* min_dists,
    void* argmin,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_distance_row_min(
    const void* distances,
    size_t m,
    size_t n,
    void* mins,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_distance_row_argmin(
    const void* distances,
    size_t m,
    size_t n,
    void* argmins,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_distance_row_topk(
    const void* distances,
    size_t m,
    size_t n,
    size_t k,
    int largest,
    void* values,
    void* indices,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_normalize_l2(
    void* data,
    size_t n_vectors,
    size_t dims,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_compute_norms(
    const void* data,
    size_t n_vectors,
    size_t dims,
    void* norms,
    NavatalaGpuQueue* queue);

// ============================================================================
// K-Means Functions
// ============================================================================

NavatalaErrorCode navatala_kmeans_fit(
    NavatalaGpuContext* ctx,
    const void* data,
    size_t n_samples,
    size_t dims,
    size_t n_clusters,
    size_t max_iters,
    float tol,
    int init_method,
    uint64_t seed,
    int metric,
    void* centroids,
    float* inertia,
    size_t* n_iter,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_kmeans_predict(
    NavatalaGpuContext* ctx,
    const void* centroids,
    size_t n_clusters,
    size_t dims,
    const void* data,
    size_t n_samples,
    int metric,
    void* labels,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_kmeans_transform(
    NavatalaGpuContext* ctx,
    const void* centroids,
    size_t n_clusters,
    size_t dims,
    const void* data,
    size_t n_samples,
    int metric,
    void* distances,
    NavatalaGpuQueue* queue);

// ============================================================================
// Neural Operators (Experimental - )
// ============================================================================

// Opaque handle types for neural operators
typedef struct NavatalaFnoLayer NavatalaFnoLayer;
typedef struct NavatalaPINOLayer NavatalaPINOLayer;
typedef struct NavatalaMeshGraphNet NavatalaMeshGraphNet;

// FNO configuration (mirrors C++ fno_config)
typedef struct NavatalaFnoConfig {
    const size_t* input_shape;
    size_t input_shape_len;
    const size_t* output_shape;
    size_t output_shape_len;
    size_t n_modes;
    size_t hidden_size;
    size_t n_layers;
    int activation;     // activation enum
    int padding;        // padding enum
    int factorized;
    int skip_connections;
    int weight_init;
    int dtype;
} NavatalaFnoConfig;

// PINO configuration
typedef struct NavatalaPINOConfig {
    const size_t* input_shape;
    size_t input_shape_len;
    const size_t* output_shape;
    size_t output_shape_len;
    int pde_type;
    size_t n_collocation_points;
    float physics_loss_weight;
    float data_loss_weight;
    float viscosity;
    float diffusivity;
    float wave_speed;
    int periodic_bc;
    float bc_loss_weight;
    size_t n_modes;
    size_t hidden_size;
    size_t n_layers;
    int activation;
} NavatalaPINOConfig;

// MGN configuration
typedef struct NavatalaMGNConfig {
    const size_t* node_features;
    size_t node_features_len;
    const size_t* edge_features;
    size_t edge_features_len;
    size_t hidden_size;
    size_t output_size;
    size_t n_layers;
    int use_edge_features;
    int residual_connections;
    int layer_norm;
    int activation;
    int aggregation;
    int dtype;
} NavatalaMGNConfig;

// FNO Functions
NavatalaErrorCode navatala_fno_create(
    NavatalaGpuContext* ctx,
    const NavatalaFnoConfig* config,
    NavatalaFnoLayer** out_layer);

NavatalaErrorCode navatala_fno_destroy(
    NavatalaFnoLayer* layer);

NavatalaErrorCode navatala_fno_forward(
    NavatalaFnoLayer* layer,
    const void* input,
    size_t batch_size,
    void* output,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_fno_forward_with_params(
    NavatalaFnoLayer* layer,
    const void* input,
    const void* params,
    size_t batch_size,
    void* output,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_fno_save(
    NavatalaFnoLayer* layer,
    const char* path);

NavatalaErrorCode navatala_fno_load(
    NavatalaGpuContext* ctx,
    const char* path,
    const NavatalaFnoConfig* config,
    NavatalaFnoLayer** out_layer);

NavatalaErrorCode navatala_fno_num_params(
    NavatalaFnoLayer* layer,
    size_t* out_num_params);

// PINO Functions
NavatalaErrorCode navatala_pino_create(
    NavatalaGpuContext* ctx,
    const NavatalaPINOConfig* config,
    NavatalaPINOLayer** out_layer);

NavatalaErrorCode navatala_pino_destroy(
    NavatalaPINOLayer* layer);

NavatalaErrorCode navatala_pino_forward(
    NavatalaPINOLayer* layer,
    const void* input,
    size_t batch_size,
    void* output,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_pino_train_step(
    NavatalaPINOLayer* layer,
    const void* input,
    const void* ground_truth,
    size_t batch_size,
    float* total_loss,
    float* data_loss,
    float* physics_loss,
    float* bc_loss,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_pino_compute_residual(
    NavatalaPINOLayer* layer,
    const void* solution,
    size_t batch_size,
    float* mean_residual,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_pino_save(
    NavatalaPINOLayer* layer,
    const char* path);

NavatalaErrorCode navatala_pino_load(
    NavatalaGpuContext* ctx,
    const char* path,
    const NavatalaPINOConfig* config,
    NavatalaPINOLayer** out_layer);

// MGN Functions
NavatalaErrorCode navatala_mgn_create(
    NavatalaGpuContext* ctx,
    const NavatalaMGNConfig* config,
    NavatalaMeshGraphNet** out_mgn);

NavatalaErrorCode navatala_mgn_destroy(
    NavatalaMeshGraphNet* mgn);

NavatalaErrorCode navatala_mgn_forward(
    NavatalaMeshGraphNet* mgn,
    const void* node_features,
    size_t n_nodes,
    const void* edge_index,
    size_t n_edges,
    const void* edge_features,
    void* output,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_mgn_forward_no_edge_features(
    NavatalaMeshGraphNet* mgn,
    const void* node_features,
    size_t n_nodes,
    const void* edge_index,
    size_t n_edges,
    void* output,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_mgn_build_edge_index(
    NavatalaGpuContext* ctx,
    const void* faces,
    size_t n_faces,
    size_t vertices_per_face,
    void* edge_index,
    size_t* n_edges,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_mgn_compute_edge_features(
    NavatalaGpuContext* ctx,
    const void* node_positions,
    size_t n_nodes,
    const void* edge_index,
    size_t n_edges,
    void* edge_features,
    NavatalaGpuQueue* queue);

NavatalaErrorCode navatala_mgn_save(
    NavatalaMeshGraphNet* mgn,
    const char* path);

NavatalaErrorCode navatala_mgn_load(
    NavatalaGpuContext* ctx,
    const char* path,
    const NavatalaMGNConfig* config,
    NavatalaMeshGraphNet** out_mgn);

NavatalaErrorCode navatala_mgn_num_params(
    NavatalaMeshGraphNet* mgn,
    size_t* out_num_params);

// ============================================================================
// Profiling Functions ()
// ============================================================================

/**
 * Get elapsed time between two recorded GPU events.
 * @param start The start event (must be recorded and completed).
 * @param end The end event (must be recorded and completed).
 * @param elapsed_ms Output for elapsed time in milliseconds.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_event_elapsed_ms(
    NavatalaGpuEvent* start,
    NavatalaGpuEvent* end,
    float* elapsed_ms);

/**
 * Push a profiling marker onto the GPU timeline.
 * @param ctx The GPU context.
 * @param name The marker name.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_profiler_push_marker(
    NavatalaGpuContext* ctx,
    const char* name);

/**
 * Pop the most recent profiling marker.
 * @param ctx The GPU context.
 * @return NAVATALA_SUCCESS on success, error code on failure.
 */
NavatalaErrorCode navatala_profiler_pop_marker(
    NavatalaGpuContext* ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NAVATALA_FFI_H
