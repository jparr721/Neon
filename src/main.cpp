#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <filesystem>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <iostream>
#include <solvers/LinearElastic.h>
#include <string>

int main() {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    igl::readOBJ("../Assets/armadillo.obj", V, F);
    igl::opengl::glfw::Viewer viewer;

    viewer.data().set_mesh(V, F);
    viewer.launch();

    LinearElastic e;
    e.Solve();

    return 0;
}
