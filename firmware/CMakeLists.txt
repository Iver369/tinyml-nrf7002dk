cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(firmware)

# --- 1. SET VARIABLES ---
set(TFLM_ROOT "${ZEPHYR_BASE}/../modules/tflite-micro")
set(CMSIS_ROOT "${ZEPHYR_BASE}/../modules/lib/cmsis-dsp")
set(CMSIS_NN_ROOT "${ZEPHYR_BASE}/../modules/lib/cmsis-nn") 
set(CMSIS_CORE "${ZEPHYR_BASE}/../modules/hal/cmsis")

# --- 2. ADD INCLUDE PATHS ---
target_include_directories(app PRIVATE 
    include
    "${TFLM_ROOT}"
    "${TFLM_ROOT}/third_party_static/flatbuffers/include"
    "${TFLM_ROOT}/third_party_static/gemmlowp"
    "${TFLM_ROOT}/third_party_static/kissfft"
    "${TFLM_ROOT}/third_party_static/ruy"
    "${TFLM_ROOT}/third_party_static/ruy/include"
    "${CMSIS_ROOT}/Include"
    "${CMSIS_NN_ROOT}"         
    "${CMSIS_NN_ROOT}/Include" 
    "${CMSIS_CORE}/CMSIS/Core/Include"
    "${CMSIS_CORE}/CMSIS/DSP/Include"
)

# --- 3. GATHER TENSORFLOW SOURCES ---
file(GLOB_RECURSE TFLM_SOURCES 
    "${TFLM_ROOT}/tensorflow/lite/micro/*.cc"
    "${TFLM_ROOT}/tensorflow/lite/micro/*.c"
    "${TFLM_ROOT}/tensorflow/lite/kernels/*.cc"
    "${TFLM_ROOT}/tensorflow/lite/kernels/*.c"
    "${TFLM_ROOT}/tensorflow/lite/kernels/internal/*.cc"
    "${TFLM_ROOT}/tensorflow/lite/kernels/internal/*.c"
    "${TFLM_ROOT}/tensorflow/lite/core/*.cc"
    "${TFLM_ROOT}/tensorflow/lite/core/*.c"
    "${TFLM_ROOT}/tensorflow/lite/schema/*.cc"
)

# --- 4. CLEANUP & EXCLUDES ---
list(FILTER TFLM_SOURCES EXCLUDE REGEX "test.cc$")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "mock")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "bench")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "_test_")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "integration_tests")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "examples")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "test_data_generation") 
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/tools/")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/python/")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/downloads/")

# --- 5. OPTIMIZATION KERNEL SELECTION ---
# Exclude Reference Kernels where optimized versions exist
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/add.cc")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/conv.cc")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/depthwise_conv.cc")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/fully_connected.cc")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/pooling.cc")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/softmax.cc")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/mul.cc")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/svdf.cc")

# <--- NEW EXCLUDES ADDED HERE TO FIX LINKER ERRORS --->
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/transpose_conv.cc")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/unidirectional_sequence_lstm.cc")

# Exclude Other Platforms
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/ethos_u/")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/hexagon/")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/xtensa/")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/arc_")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "/kernels/ceva/")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "ceva")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "arc_")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "bluepill")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "chre")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "riscv")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "xtensa")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "apollo3")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "stm32")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "himax")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "esp")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "hexagon")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "cortex_m_corstone_300")
list(FILTER TFLM_SOURCES EXCLUDE REGEX "cortex_m_generic")

# --- 6. TARGET DEFINITIONS ---
target_sources(app PRIVATE 
    src/main.c 
    src/dsp.c 
    src/inference.cpp
    src/model.cpp
    src/data.c
    ${TFLM_SOURCES}
)

target_compile_definitions(app PRIVATE 
    CMSIS_NN 
    TF_LITE_STATIC_MEMORY
)