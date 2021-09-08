// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <solvers/materials/Homogenization.h>
#include <visualizer/controllers/SolverController.h>

visualizer::controllers::SolverController::SolverController(const int dim, const int void_dim, const int thickness) {
    material_ = solvers::materials::OrthotropicMaterial(E_baseline, v_baseline, G_baseline);
    ReloadMeshes(dim, void_dim, thickness);
}

void visualizer::controllers::SolverController::ReloadMeshes(const int dim, const int void_dim, const int thickness) {
    ComputeUniformMesh(dim);
    ComputeVoidMesh(dim, void_dim, thickness);
}

void visualizer::controllers::SolverController::ComputeUniformMesh(const int dim) {
    NEON_LOG_INFO("Recomputing uniform mesh");
    auto gen = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(dim, dim, dim);

    MatrixXr V;
    MatrixXi F;
    gen->GenerateImplicitFunctionBasedMaterial(meshing::ImplicitSurfaceGenerator<Real>::kNoThickness, V, F);

    // Now re-make the mesh object
    uniform_mesh_ = std::make_shared<meshing::Mesh>(V, F, tetgen_flags);
}

void visualizer::controllers::SolverController::ComputeVoidMesh(int dim, int void_dim, int thickness) {
    NEON_LOG_INFO("Recomputing void mesh");
    auto gen = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(
            dim, dim, dim, meshing::ImplicitSurfaceGenerator<Real>::Behavior::kIsotropic,
            meshing::ImplicitSurfaceGenerator<Real>::MakeInclusion(1, void_dim));

    MatrixXr V;
    MatrixXi F;
    gen->GenerateImplicitFunctionBasedMaterial(thickness, V, F);
    perforated_surface_mesh_ = gen->Surface();

    // Now re-make the mesh object
    perforated_mesh_ = std::make_shared<meshing::Mesh>(V, F, tetgen_flags);
}

void visualizer::controllers::SolverController::HomogenizeVoidMesh(const solvers::materials::Material &material) {
    NEON_LOG_INFO("Homogenizing void mesh");
    auto homogenization = std::make_unique<solvers::materials::Homogenization>(perforated_surface_mesh_, material);
    homogenization->Solve();
    material_ = homogenization->Coefficients();
}
auto visualizer::controllers::SolverController::UniformSolver() -> std::shared_ptr<solvers::fem::LinearElastic> & {
    return uniform_solver_;
}
auto visualizer::controllers::SolverController::PerforatedSolver() -> std::shared_ptr<solvers::fem::LinearElastic> & {
    return perforated_solver_;
}
auto visualizer::controllers::SolverController::UniformMesh() -> std::shared_ptr<meshing::Mesh> & {
    return uniform_mesh_;
}
auto visualizer::controllers::SolverController::PerforatedMesh() -> std::shared_ptr<meshing::Mesh> & {
    return perforated_mesh_;
}
