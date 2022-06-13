cmake_minimum_required(VERSION 3.16)

project(model)

file(GLOB project_file main.cpp)

add_executable(${PROJECT_NAME} ${project_file})

target_link_libraries(
                        ${PROJECT_NAME} 
                        ${CGLIBS} 
                        glad
)
add_executable(depth2 depth2.cpp)

target_link_libraries(
                        depth2
                        ${CGLIBS} 
                        glad
)

add_executable(depth_view depth_view.cpp)

target_link_libraries(
                        depth_view
                        ${CGLIBS} 
                        glad
)

add_executable(stencil_testing stencil_testing.cpp)

target_link_libraries(
                        stencil_testing
                        ${CGLIBS} 
                        glad
)
