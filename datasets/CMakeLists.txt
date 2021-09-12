project(datasets)

add_library(${PROJECT_NAME} SHARED
        src/Deformation.cpp include/datasets/Deformation.h src/Homogenization.cpp include/datasets/Homogenization.h src/SolverMask.cpp include/datasets/SolverMask.h)

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

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen utilities solvers meshing)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

add_executable(
        ${PROJECT_NAME}DeformationDatasetTest
        tests/DeformationTest.cpp
)

target_link_libraries(
        ${PROJECT_NAME}DeformationDatasetTest
        LINK_PUBLIC ${Boost_LIBRARIES}
        ${PROJECT_NAME}
        solvers
        meshing
        utilities
)