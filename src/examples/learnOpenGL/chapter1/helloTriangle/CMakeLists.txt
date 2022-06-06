cmake_minimum_required(VERSION 3.16)

project(helloTriangle)

file(GLOB project_file main.cpp)

add_executable(${PROJECT_NAME} ${project_file})

target_link_libraries(
                        ${PROJECT_NAME} 
                        ${CGLIBS} 
                        glad
)

add_executable(triangleV1 triangleV1.cpp)

target_link_libraries(
                        triangleV1
                        ${CGLIBS} 
                        glad
)

add_executable(triangleV2 triangleV2.cpp)

target_link_libraries(
                        triangleV2
                        ${CGLIBS} 
                        glad
)

add_executable(triangleV3 triangleV3.cpp)

target_link_libraries(
                        triangleV3
                        ${CGLIBS} 
                        glad
)