cmake_minimum_required(VERSION 3.19)
project(Neon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          ON)
include(libigl)

add_executable(Neon main.cpp)

target_link_libraries(Neon PRIVATE Eigen3::Eigen igl::core igl::opengl_glfw)