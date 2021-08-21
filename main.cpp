#include <iostream>
#include <Eigen/Dense>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>

int main() {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    igl::readOBJ("../armadillo.obj", V, F);
    igl::opengl::glfw::Viewer viewer;

    viewer.data().set_mesh(V, F);
    viewer.launch();

    return 0;
}
