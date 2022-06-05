cmake_minimum_required(VERSION 3.16)

project(useBoost)

file(GLOB project_file main.cpp)

add_executable(${PROJECT_NAME} ${project_file})

target_link_libraries(${PROJECT_NAME} -lpthread
)