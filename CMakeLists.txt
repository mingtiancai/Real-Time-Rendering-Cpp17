cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

find_package(glfw3 REQUIRED)
message("Found GLFW3 in " ${glfw3_INCLUDE_DIRS})

INCLUDE_DIRECTORIES(${glfw3_INCLUDE_DIRS})
MESSAGE(STATUS "FIND GLFW_INCLUDE_DIRS " ${glfw3_INCLUDE_DIRS})

project(show)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_RELEASE_OUTPUT_DIR})


file(GLOB project_file glad.c main.cpp)

add_executable(${PROJECT_NAME} ${project_file})

set(CGLIBS 
    -lglfw3
    -lGL
    -lm
    -lXrandr
    -lXi
    -lX11
    -lXxf86vm
    -lpthread
    -ldl
    -lXinerama
    -lXcursor
)

target_link_libraries(${PROJECT_NAME} ${CGLIBS}
)
