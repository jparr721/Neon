#include <Eigen/Core>
#include <filesystem>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <utilities/math/LinearAlgebra.h>

int main() {
    MatrixXr V;
    MatrixX<int> F;

    igl::readOBJ("../Assets/armadillo.obj", V, F);
    igl::opengl::glfw::Viewer viewer;

    viewer.data().set_mesh(V, F);
    viewer.launch();

    return 0;
}
