cmake_minimum_required(VERSION 3.19)
project(Neon LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_BINARY_DIR ${CMAKE_BUILD_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(INSTALL_LIB_DIR lib CACHE PATH "Install directory for library code")
set(INSTALL_BIN_DIR CACHE PATH "Install directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Install directory for header files")

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(LIBIGL_WITH_OPENGL "Use OpenGL" ON)
option(LIBIGL_WITH_OPENGL_GLFW "Use GLFW" ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui" ON)
include(libigl)

if (MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif ()

foreach (p LIB BIN INCLUDE CMAKE)
    set(var INSTALL_${p}_DIR)
    if (NOT IS_ABSOLUTE "${${var}}")
        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif ()
endforeach ()

if (USE_DOUBLE)
    add_definitions(-DNEON_USE_DOUBLE)
endif (USE_DOUBLE)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
include_directories("${PROJECT_SOURCE_DIR}"
        "${PROJECT_BINARY_DIR}"
        ${Boost_INCLUDE_DIRS})

#add_executable(Neon src/main.cpp)
#
#target_include_directories(${PROJECT_NAME} PUBLIC
#        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}}/include>
#        $<INSTALL_INTERFACE:include>)
#
#target_link_libraries(Neon PRIVATE
#        Eigen3::Eigen
#        igl::core
#        igl::opengl_glfw
#        solvers
#        utilities)
#
#file(COPY ${CMAKE_SOURCE_DIR}/Assets DESTINATION ${CMAKE_BINARY_DIR})

add_subdirectory(solvers)
add_subdirectory(utilities)
add_subdirectory(meshing)
add_subdirectory(visualizer)