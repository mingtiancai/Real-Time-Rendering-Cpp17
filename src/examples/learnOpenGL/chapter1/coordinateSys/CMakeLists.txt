cmake_minimum_required(VERSION 3.16)

project(coordinateSys)

file(GLOB project_file main.cpp)

add_executable(${PROJECT_NAME} ${project_file})

target_link_libraries(
                        ${PROJECT_NAME} 
                        ${CGLIBS} 
                        glad
)

add_executable(depth depth.cpp)

target_link_libraries(
                        depth
                        ${CGLIBS} 
                        glad
)

add_executable(multiple multiple.cpp)

target_link_libraries(
                        multiple
                        ${CGLIBS} 
                        glad
)