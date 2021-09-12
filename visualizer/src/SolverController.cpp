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
#include <utilities/math/Geometry.h>
#include <visualizer/controllers/SolverController.h>

visualizer::controllers::SolverController::SolverController(const int dim, const int void_dim, const int thickness) {
    // Default parameter set.
    constexpr Real E_baseline = 30000;
    constexpr Real v_baseline = 0.3;
    constexpr Real G_baseline = E_baseline / (2 * (1 + v_baseline));
    material_ = solvers::materials::OrthotropicMaterial(E_baseline, v_baseline, G_baseline);
    ReloadMeshes(dim, void_dim, thickness);
}

void visualizer::controllers::SolverController::ReloadMeshes(const int dim, const int void_dim, const int thickness) {
    ComputeUniformMesh(dim);
    ComputeVoidMesh(dim, void_dim, thickness);
}

void visualizer::controllers::SolverController::ComputeUniformMesh(const int dim) {
    solvers_need_reload = true;
    NEON_LOG_INFO("Recomputing uniform mesh");
    auto gen = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(dim, dim, dim);

    MatrixXr V;
    MatrixXi F;
    gen->GenerateImplicitFunctionBasedMaterial(meshing::ImplicitSurfaceGenerator<Real>::kNoThickness, V, F);

    // Now re-make the mesh object
    uniform_mesh_ = std::make_shared<meshing::Mesh>(V, F, tetgen_flags);
}

void visualizer::controllers::SolverController::ComputeVoidMesh(int dim, int void_dim, int thickness) {
    solvers_need_reload = true;
    NEON_LOG_INFO("Recomputing void mesh");
    auto gen = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(
            dim, dim, dim, meshing::ImplicitSurfaceGenerator<Real>::Behavior::kIsotropic,
            meshing::ImplicitSurfaceGenerator<Real>::MakeInclusion(1, void_dim));

    MatrixXr V;
    MatrixXi F;
    gen->GenerateImplicitFunctionBasedMaterial(thickness, V, F);
    NEON_LOG_INFO(V.size());
    NEON_LOG_INFO(F.size());
    perforated_surface_mesh_ = gen->Surface();

    // Now re-make the mesh object
    perforated_mesh_ = std::make_shared<meshing::Mesh>(V, F, tetgen_flags);

    utilities::math::Scoot(Vector3r(uniform_mesh_->positions.col(0).maxCoeff() * 2, 0, 0),
                           perforated_mesh_->rest_positions);
    perforated_mesh_->ResetMesh();
}

void visualizer::controllers::SolverController::HomogenizeVoidMesh() {
    solvers_need_reload = true;
    NEON_LOG_INFO("Homogenizing void mesh");
    auto homogenization = std::make_unique<solvers::materials::Homogenization>(
            perforated_surface_mesh_,
            solvers::materials::MaterialFromLameCoefficients(1, "m", approximate_mu_, approximate_lambda_));
    homogenization->Solve();
    material_ = homogenization->Coefficients();
}

auto visualizer::controllers::SolverController::UniformMesh() const -> const std::shared_ptr<meshing::Mesh> & {
    return uniform_mesh_;
}
auto visualizer::controllers::SolverController::PerforatedMesh() const -> const std::shared_ptr<meshing::Mesh> & {
    return perforated_mesh_;
}

void visualizer::controllers::SolverController::ResetMeshPositions() {
    uniform_mesh_->ResetMesh();
    perforated_mesh_->ResetMesh();
}

void visualizer::controllers::SolverController::ReloadSolvers(solvers::fem::LinearElastic::Type type) {
    solvers_need_reload = false;

    const Vector3r force(0, force_, 0);

    uniform_interior_nodes_.clear();
    uniform_force_nodes_.clear();
    uniform_fixed_nodes_.clear();
    uniform_boundary_conditions_.clear();
    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, uniform_mesh_, uniform_interior_nodes_,
                                 uniform_force_nodes_, uniform_fixed_nodes_);
    solvers::boundary_conditions::LoadBoundaryConditions(force, uniform_mesh_, uniform_force_nodes_,
                                                         uniform_interior_nodes_, uniform_boundary_conditions_);
    uniform_solver_ =
            std::make_unique<solvers::fem::LinearElastic>(uniform_boundary_conditions_, material_, uniform_mesh_, type);

    perforated_interior_nodes_.clear();
    perforated_force_nodes_.clear();
    perforated_fixed_nodes_.clear();
    perforated_boundary_conditions_.clear();
    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, perforated_mesh_, perforated_interior_nodes_,
                                 perforated_force_nodes_, perforated_fixed_nodes_);
    solvers::boundary_conditions::LoadBoundaryConditions(force, perforated_mesh_, perforated_force_nodes_,
                                                         perforated_interior_nodes_, perforated_boundary_conditions_);
    perforated_solver_ = std::make_unique<solvers::fem::LinearElastic>(perforated_boundary_conditions_, material_,
                                                                       perforated_mesh_, type);

    if (type == solvers::fem::LinearElastic::Type::kDynamic) {
        uniform_integrator_ = std::make_unique<solvers::integrators::CentralDifferenceMethod>(
                dt_, mass_, uniform_solver_->K_e, uniform_solver_->U_e, uniform_solver_->F_e);
        perforated_integrator_ = std::make_unique<solvers::integrators::CentralDifferenceMethod>(
                dt_, mass_, perforated_solver_->K_e, perforated_solver_->U_e, perforated_solver_->F_e);
    }
}

void visualizer::controllers::SolverController::SolveUniform(const bool dynamic) {
    NEON_ASSERT_ERROR(uniform_solver_ != nullptr, "Uniform solver has not been initialized");
    if (dynamic) {
        // Make sure the integrator has been initialized.
        NEON_ASSERT_ERROR(uniform_integrator_ != nullptr, "Uniform integrator has not been initialized.");
        uniform_integrator_->Solve(uniform_solver_->F_e, uniform_solver_->U_e);
    }

    uniform_solver_->Solve(uniform_displacements, uniform_stresses);
    uniform_mesh_->Update(uniform_displacements);
}

void visualizer::controllers::SolverController::SolvePerforated(const bool dynamic) {
    NEON_ASSERT_ERROR(perforated_solver_ != nullptr, "Perforated solver has not been initialized");
    if (dynamic) {
        // Make sure the integrator has been initialized.
        NEON_ASSERT_ERROR(perforated_integrator_ != nullptr, "Perforated integrator has not been initialized.");
        perforated_integrator_->Solve(perforated_solver_->F_e, perforated_solver_->U_e);
    }
    perforated_solver_->Solve(perforated_displacements, perforated_stresses);
    perforated_mesh_->Update(perforated_displacements);
}
