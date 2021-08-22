project(visualizer)

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}}/include>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME} PRIVATE
        Eigen3::Eigen
        igl::core
        igl::opengl_glfw
        solvers
        utilities)

file(COPY ${CMAKE_SOURCE_DIR}/Assets DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})