cmake_minimum_required(VERSION 3.16)

project(textures)

file(GLOB project_file main.cpp)

add_executable(${PROJECT_NAME} ${project_file})

target_link_libraries(
                        ${PROJECT_NAME} 
                        ${CGLIBS} 
                        glad
)

add_executable(mix mix.cpp)

target_link_libraries(
                        mix
                        ${CGLIBS} 
                        glad
)