project(solvers)

add_library(${PROJECT_NAME} SHARED
        include/solvers/FEM/LinearElastic.h
        include/solvers/materials/Homogenization.h

        src/LinearElastic.cpp
        src/Homogenization.cpp
        include/solvers/materials/Material.h src/Material.cpp
        include/solvers/utilities/BoundaryCondition.h
        src/BoundaryCondition.cpp
        src/CentralDifferenceMethod.cpp
        include/solvers/integrators/CentralDifferenceMethod.h include/solvers/materials/OrthotropicMaterial.h src/OrthotropicMaterial.cpp include/solvers/materials/ScalarField.h)

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

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen igl::core igl::opengl_glfw utilities meshing)

install(DIRECTORY include/ DESTINATION "${INSTALL_INCLUDE_DIR}")
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

add_executable(
        ${PROJECT_NAME}HomogenizationTest
        tests/HomogenizationTest.cpp
)

target_link_libraries(
        ${PROJECT_NAME}HomogenizationTest
        LINK_PUBLIC ${Boost_LIBRARIES}
        ${PROJECT_NAME}
        utilities
        meshing
)

add_executable(
        ${PROJECT_NAME}LinearElasticTest
        tests/LinearElasticTest.cpp
)

target_link_libraries(
        ${PROJECT_NAME}LinearElasticTest
        LINK_PUBLIC ${Boost_LIBRARIES}
        ${PROJECT_NAME}
        utilities
        meshing
)

add_executable(
        ${PROJECT_NAME}CentralDifferenceMethodTest
        tests/CentralDifferenceMethodTest.cpp
)

target_link_libraries(
        ${PROJECT_NAME}CentralDifferenceMethodTest
        LINK_PUBLIC ${Boost_LIBRARIES}
        ${PROJECT_NAME}
        utilities
)