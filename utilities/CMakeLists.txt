project(utilities)

add_library(${PROJECT_NAME} SHARED src/LinearAlgebra.cpp include/runtime/NeonAssert.h)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        PRIVATE
        ${PROJECT_SOURCE_DIR}/src)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

add_executable(
        ${PROJECT_NAME}AssertionTest
        tests/AssertTest.cpp
)

target_link_libraries(
        ${PROJECT_NAME}AssertionTest
        LINK_PUBLIC ${Boost_LIBRARIES}
        ${PROJECT_NAME}
)
