# SPDX-License-Identifier: Apache-2.0
nanobind_add_module(navatala_gpu_ext navatala_gpu_ext.cpp)
target_compile_features(navatala_gpu_ext PRIVATE cxx_std_20)
target_include_directories(navatala_gpu_ext PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NAVATALA_FFI_INCLUDE_DIR}
)
if(UNIX AND NOT APPLE)
    target_link_options(navatala_gpu_ext PRIVATE "LINKER:--no-as-needed")
endif()
target_link_libraries(navatala_gpu_ext PRIVATE ${NAVATALA_FFI_LIBRARY})
set_target_properties(navatala_gpu_ext PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
install(TARGETS navatala_gpu_ext LIBRARY DESTINATION navatala_gpu)
install(FILES ${NAVATALA_FFI_LIBRARY} DESTINATION navatala_gpu)
