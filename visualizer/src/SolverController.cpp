// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <meshing/DofOptimizer.h>
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

void visualizer::controllers::SolverController::ResetMeshPositions() {
    uniform_mesh_->ResetMesh();
    perforated_mesh_->ResetMesh();
}

void visualizer::controllers::SolverController::ReloadSolvers(solvers::fem::LinearElastic::Type type) {
    const auto boundary_conditions = ComputeActiveDofs();

    uniform_solver_ =
            std::make_unique<solvers::fem::LinearElastic>(boundary_conditions, material_, uniform_mesh_, type);

    if (type == solvers::fem::LinearElastic::Type::kDynamic) {
        uniform_integrator_ = std::make_unique<solvers::integrators::CentralDifferenceMethod>(
                dt_, mass_, uniform_solver_->K_e, uniform_solver_->U_e, uniform_solver_->F_e);
    }
}

auto visualizer::controllers::SolverController::ComputeActiveDofs()
        -> solvers::boundary_conditions::BoundaryConditions {
    interior_nodes_.clear();
    force_nodes_.clear();
    fixed_nodes_.clear();
    const Vector3r force(0, force_, 0);
    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, uniform_mesh_, interior_nodes_, force_nodes_,
                                 fixed_nodes_);
    solvers::boundary_conditions::BoundaryConditions all_boundary_conditions;
    solvers::boundary_conditions::LoadBoundaryConditions(force, uniform_mesh_, force_nodes_, interior_nodes_,
                                                         all_boundary_conditions);
    return all_boundary_conditions;
}
auto visualizer::controllers::SolverController::UniformIntegrator()
        -> std::shared_ptr<solvers::integrators::CentralDifferenceMethod> & {
    return uniform_integrator_;
}
auto visualizer::controllers::SolverController::PerforatedIntegrator()
        -> std::shared_ptr<solvers::integrators::CentralDifferenceMethod> & {
    return perforated_integrator_;
}
