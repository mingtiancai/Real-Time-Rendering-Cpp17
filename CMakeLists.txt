cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

find_package(Boost REQUIRED)

if(Boost_FOUND)
	MESSAGE( STATUS "Boost_INCLUDE_DIRS = " ${Boost_INCLUDE_DIRS})
    MESSAGE( STATUS "Boost_LIBRARIES = " ${Boost_LIBRARIES})
    MESSAGE( STATUS "Boost_LIB_VERSION = " ${Boost_LIB_VERSION})
    MESSAGE( STATUS "Boost_LIBRARY_DIRS = " ${Boost_LIBRARY_DIRS})
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    # target_link_libraries (testBoost ${Boost_LIBRARIES})
endif()

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
    -lassimp
)

add_subdirectory(src)