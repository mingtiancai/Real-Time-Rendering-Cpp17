cmake_minimum_required(VERSION 3.16)

set(RTRC_DIR ${PROJECT_SOURCE_DIR})

include_directories(${RTRC_DIR}/src/3rd)
message("include: " ${RTRC_DIR}/src/3rd)

add_subdirectory(3rd)
add_subdirectory(examples)