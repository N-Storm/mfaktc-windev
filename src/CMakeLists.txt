cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mfaktc LANGUAGES C CUDA)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -std=gnu11 -flto -ffunction-sections -fdata-sections -ffat-lto-objects -Wall -Wextra -malign-double -Wl,--warn-common -Wl,--gc-sections")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Og -std=gnu11 -malign-double -Wall -Wextra -Wl,--warn-common")
endif()


add_executable(mfaktc STATIC
  sieve.c timer.c parse.c read_config.c mfaktc.c checkpoint.c 
  signal_handler.c output.c
  tf_72bit.cu tf_96bit.cu tf_barrett96.cu tf_barrett96_gs.cu gpusieve.cu
  )

set_target_properties( mfaktc PROPERTIES CUDA_SEPARABLE_COMPILATION ON )

message(STATUS "Project: ${PROJECT_NAME} ${GIT_VERSION}")

find_package(cudart)
