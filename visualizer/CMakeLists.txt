project(visualizer)

if (CMAKE_BUILD_TYPE EQUAL "Release")
    add_compile_options(/O2)
endif ()

add_executable(${PROJECT_NAME} src/main.cpp include/visualizer/Visualizer.h src/Visualizer.cpp src/SolverController.cpp include/visualizer/controllers/SolverController.h)

target_include_directories(${PROJECT_NAME} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_link_libraries(${PROJECT_NAME} PRIVATE
        Eigen3::Eigen
        igl::core
        igl::opengl
        igl::opengl_glfw
        igl::opengl_glfw_imgui
        OpenMP::OpenMP_CXX
        meshing
        solvers
        utilities
        datasets
        "${TORCH_LIBRARIES}")

if (MSVC AND USE_NEURAL_NETWORK)
    # Pytorch - Copy DLLs to avoid memory errors
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)

file(COPY ${CMAKE_SOURCE_DIR}/Assets DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})