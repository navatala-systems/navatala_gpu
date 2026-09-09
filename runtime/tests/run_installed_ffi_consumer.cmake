foreach(required_var RUNTIME_BUILD_DIR RUNTIME_SOURCE_DIR INSTALL_PREFIX CONSUMER_BUILD_DIR)
    if(NOT DEFINED ${required_var})
        message(FATAL_ERROR "Missing required variable: ${required_var}")
    endif()
endforeach()

file(REMOVE_RECURSE "${INSTALL_PREFIX}" "${CONSUMER_BUILD_DIR}")

set(install_command
    "${CMAKE_COMMAND}" --install "${RUNTIME_BUILD_DIR}" --prefix "${INSTALL_PREFIX}")
if(DEFINED TEST_CONFIG AND NOT TEST_CONFIG STREQUAL "")
    list(APPEND install_command --config "${TEST_CONFIG}")
endif()
execute_process(
    COMMAND ${install_command}
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR
        "gpu_runtime install failed (${install_result})\n${install_output}\n${install_error}")
endif()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${RUNTIME_SOURCE_DIR}/tests/installed_ffi_consumer"
    -B "${CONSUMER_BUILD_DIR}"
    "-DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}")
if(DEFINED RUNTIME_CXX_FLAGS AND NOT RUNTIME_CXX_FLAGS STREQUAL "")
    list(APPEND configure_command "-DCMAKE_C_FLAGS=${RUNTIME_CXX_FLAGS}")
endif()
if(DEFINED RUNTIME_EXE_LINKER_FLAGS AND NOT RUNTIME_EXE_LINKER_FLAGS STREQUAL "")
    list(APPEND configure_command
        "-DCMAKE_EXE_LINKER_FLAGS=${RUNTIME_EXE_LINKER_FLAGS}")
endif()
if(DEFINED TEST_CONFIG AND NOT TEST_CONFIG STREQUAL "")
    list(APPEND configure_command "-DCMAKE_BUILD_TYPE=${TEST_CONFIG}")
endif()
execute_process(
    COMMAND ${configure_command}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
        "installed consumer configure failed (${configure_result})\n${configure_output}\n${configure_error}")
endif()

set(build_command "${CMAKE_COMMAND}" --build "${CONSUMER_BUILD_DIR}")
if(DEFINED TEST_CONFIG AND NOT TEST_CONFIG STREQUAL "")
    list(APPEND build_command --config "${TEST_CONFIG}")
endif()
execute_process(
    COMMAND ${build_command}
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR
        "installed consumer build failed (${build_result})\n${build_output}\n${build_error}")
endif()
