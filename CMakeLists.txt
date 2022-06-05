cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

find_package(glfw3 REQUIRED)

project(REAL-TIME-RENDERING-CPP17)

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

add_subdirectory(src)