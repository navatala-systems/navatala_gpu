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

#include <navatala/navatala_ffi.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { element_count = 64 };

static char* duplicate_string(const char* value)
{
    size_t length = 0;
    char* copy = NULL;
    if (!value) {
        return NULL;
    }
    length = strlen(value) + 1U;
    copy = (char*)malloc(length);
    if (copy) {
        memcpy(copy, value, length);
    }
    return copy;
}

static float absolute_difference(float lhs, float rhs)
{
    float difference = lhs - rhs;
    return difference < 0.0F ? -difference : difference;
}

static int set_backend_environment(const char* value)
{
#if defined(_WIN32)
    return _putenv_s("GPU_RUNTIME_BACKEND", value ? value : "");
#else
    return value ? setenv("GPU_RUNTIME_BACKEND", value, 1)
                 : unsetenv("GPU_RUNTIME_BACKEND");
#endif
}

static unsigned char* read_binary(const char* path, size_t* size)
{
    FILE* file = fopen(path, "rb");
    long length = 0;
    unsigned char* bytes = NULL;

    if (!file || fseek(file, 0, SEEK_END) != 0) {
        if (file) {
            fclose(file);
        }
        return NULL;
    }
    length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    bytes = (unsigned char*)malloc((size_t)length);
    if (!bytes || fread(bytes, 1, (size_t)length, file) != (size_t)length) {
        free(bytes);
        fclose(file);
        return NULL;
    }

    fclose(file);
    *size = (size_t)length;
    return bytes;
}

static int expect_success(NavatalaErrorCode status, const char* operation)
{
    if (status == NAVATALA_SUCCESS) {
        return 1;
    }
    fprintf(stderr, "%s failed with Navatala status %d\n", operation, (int)status);
    return 0;
}

int main(int argc, char** argv)
{
    NavatalaGpuContext* context = NULL;
    NavatalaGpuQueue* queue = NULL;
    NavatalaGpuBuffer* input_buffer = NULL;
    NavatalaGpuBuffer* output_buffer = NULL;
    NavatalaGpuProgram* program = NULL;
    NavatalaGpuBuffer* arguments[2] = {NULL, NULL};
    unsigned char* spirv = NULL;
    size_t spirv_size = 0;
    float input[element_count];
    float output[element_count];
    float offset_result = 0.0F;
    const char* original_backend = getenv("GPU_RUNTIME_BACKEND");
    char* saved_backend = duplicate_string(original_backend);
    int result = 1;
    size_t index = 0;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <shader.spv>\n", argv[0]);
        result = 2;
        goto cleanup;
    }

    spirv = read_binary(argv[1], &spirv_size);
    if (!spirv) {
        fprintf(stderr, "failed to read SPIR-V fixture: %s\n", argv[1]);
        result = 2;
        goto cleanup;
    }

    if (set_backend_environment("opencl") != 0) {
        fprintf(stderr, "failed to set conflicting backend environment\n");
        result = 2;
        goto cleanup;
    }

    {
        NavatalaErrorCode status = navatala_gpu_create_context(
            NAVATALA_BACKEND_VULKAN_FFI, 0, &context);
        if (status == NAVATALA_NOT_FOUND) {
            result = 77;
            goto cleanup;
        }
        if (!expect_success(status, "navatala_gpu_create_context")) {
            goto cleanup;
        }
    }

    if (navatala_gpu_context_get_backend(context) != NAVATALA_BACKEND_VULKAN_FFI ||
        !getenv("GPU_RUNTIME_BACKEND") ||
        strcmp(getenv("GPU_RUNTIME_BACKEND"), "opencl") != 0) {
        fprintf(stderr, "explicit Vulkan selection failed or mutated process environment\n");
        goto cleanup;
    }

    if (!expect_success(navatala_gpu_create_queue(context, 0, &queue),
                        "navatala_gpu_create_queue") ||
        !expect_success(navatala_gpu_create_buffer(
                            context, sizeof(input), &input_buffer),
                        "navatala_gpu_create_buffer(input)") ||
        !expect_success(navatala_gpu_create_buffer(
                            context, sizeof(output), &output_buffer),
                        "navatala_gpu_create_buffer(output)") ||
        !expect_success(navatala_gpu_create_program(
                            context, NAVATALA_PROGRAM_SOURCE_SPIRV,
                            spirv, spirv_size, "main", &program),
                        "navatala_gpu_create_program")) {
        goto cleanup;
    }

    for (index = 0; index < element_count; ++index) {
        input[index] = (float)(index + 1U) * 0.25F;
        output[index] = -1.0F;
    }

    if (!expect_success(navatala_gpu_copy_h2d(
                            input_buffer, input, sizeof(input), queue),
                        "navatala_gpu_copy_h2d")) {
        goto cleanup;
    }

    input[7] = 9.0F;
    if (!expect_success(navatala_gpu_copy_h2d_offset(
                            input_buffer, 7U * sizeof(float), &input[7],
                            sizeof(float), queue),
                        "navatala_gpu_copy_h2d_offset")) {
        goto cleanup;
    }

    if (!expect_success(navatala_gpu_copy_d2d(
                            output_buffer, input_buffer, sizeof(input), queue),
                        "navatala_gpu_copy_d2d")) {
        goto cleanup;
    }
    output[11] = 5.0F;
    if (!expect_success(navatala_gpu_copy_h2d_offset(
                            output_buffer, 11U * sizeof(float), &output[11],
                            sizeof(float), queue),
                        "navatala_gpu_copy_h2d_offset(d2d source)") ||
        !expect_success(navatala_gpu_copy_d2d_offset(
                            input_buffer, 11U * sizeof(float), output_buffer,
                            11U * sizeof(float), sizeof(float), queue),
                        "navatala_gpu_copy_d2d_offset")) {
        goto cleanup;
    }
    input[11] = 5.0F;

    arguments[0] = input_buffer;
    arguments[1] = output_buffer;
    if (!expect_success(navatala_gpu_launch_program(
                            queue, program, arguments, 2, 1, 1, 1, 64, 1, 1),
                        "navatala_gpu_launch_program") ||
        !expect_success(navatala_gpu_queue_sync(queue),
                        "navatala_gpu_queue_sync") ||
        !expect_success(navatala_gpu_copy_d2h_offset(
                            output_buffer, 7U * sizeof(float), &offset_result,
                            sizeof(float), queue),
                        "navatala_gpu_copy_d2h_offset") ||
        !expect_success(navatala_gpu_copy_d2h(
                            output_buffer, output, sizeof(output), queue),
                        "navatala_gpu_copy_d2h")) {
        goto cleanup;
    }

    if (absolute_difference(offset_result, 81.0F) > 1.0e-6F) {
        fprintf(stderr, "offset result mismatch: %.9g\n", (double)offset_result);
        goto cleanup;
    }
    for (index = 0; index < element_count; ++index) {
        float expected = input[index] * input[index];
        if (absolute_difference(output[index], expected) > 1.0e-6F) {
            fprintf(stderr,
                    "result mismatch at %zu: expected %.9g, got %.9g\n",
                    index, (double)expected, (double)output[index]);
            goto cleanup;
        }
    }

    result = 0;

cleanup:
    navatala_gpu_destroy_program(program);
    navatala_gpu_destroy_buffer(output_buffer);
    navatala_gpu_destroy_buffer(input_buffer);
    navatala_gpu_destroy_queue(queue);
    navatala_gpu_destroy_context(context);
    free(spirv);
    if (saved_backend) {
        (void)set_backend_environment(saved_backend);
    } else {
        (void)set_backend_environment(NULL);
    }
    free(saved_backend);
    return result;
}
