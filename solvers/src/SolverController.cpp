// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <igl/collapse_small_triangles.h>
#include <igl/marching_cubes.h>
#include <meshing/DofOptimizer.h>
#include <meshing/MeshOptimizer.h>
#include <meshing/implicit_surfaces/PeriodicGyroid.h>
#include <solvers/include/solvers/controllers/SolverController.h>
#include <solvers/materials/Homogenization.h>
#include <utilities/math/Geometry.h>

solvers::controllers::SolverController::SolverController(int dim) {
    // Default parameter set.
    constexpr Real E_baseline = 30000;
    constexpr Real v_baseline = 0.3;
    // Because this is not a transverse isotropic or fully isotropic system, G must be defined on its own.
    constexpr Real G_baseline = 1;

    uniform_material_ = solvers::materials::OrthotropicMaterial(E_baseline, v_baseline, G_baseline);
    ComputeUniformMesh(dim);
}

solvers::controllers::SolverController::SolverController(const int dim, const Real amplitude, const Real thickness) {
    // Default parameter set.
    constexpr Real E_baseline = 30000;
    constexpr Real v_baseline = 0.3;
    // Because this is not a transverse isotropic or fully isotropic system, G must be defined on its own.
    constexpr Real G_baseline = 1;
    uniform_material_ = solvers::materials::OrthotropicMaterial(E_baseline, v_baseline, G_baseline);
    perforated_material_ = solvers::materials::OrthotropicMaterial(E_baseline, v_baseline, G_baseline);
    ReloadMeshes(dim, amplitude, thickness);
}

void solvers::controllers::SolverController::ReloadMeshes(const int dim, const Real amplitude, const Real thickness) {
    ComputeUniformMesh(dim);
    ComputeVoidMesh(dim, amplitude, thickness);
}

void solvers::controllers::SolverController::ComputeUniformMesh(const int dim) {
    solvers_need_reload = true;
    auto gen = std::make_unique<meshing::implicit_surfaces::ImplicitSurfaceGenerator<Real>>(dim, dim, dim);

    MatrixXr V;
    MatrixXi F;
    gen->GenerateImplicitFunctionBasedMaterial(meshing::implicit_surfaces::ImplicitSurfaceGenerator<Real>::kNoThickness,
                                               false, V, F);

    // Now re-make the mesh object
    uniform_mesh_ = std::make_shared<meshing::Mesh>(V, F, tetgen_flags);
}

void solvers::controllers::SolverController::ComputeVoidMesh(const int dim, const Real amplitude,
                                                             const Real thickness) {
    solvers_need_reload = true;
    MatrixXr V;
    MatrixXi F;
    meshing::implicit_surfaces::ComputeImplicitGyroidMarchingCubes(
            amplitude, thickness, dim, meshing::implicit_surfaces::SineFunction, V, F, perforated_surface_mesh_);

    perforated_mesh_ = std::make_shared<meshing::Mesh>(V, F, tetgen_flags);

    solvers::math::Scoot(Vector3r(uniform_mesh_->positions.col(0).maxCoeff() * 2, 0, 0),
                         perforated_mesh_->rest_positions);
    perforated_mesh_->ResetMesh();
}

void solvers::controllers::SolverController::HomogenizeVoidMesh() {
    solvers_need_reload = true;
    auto homogenization = std::make_unique<solvers::materials::Homogenization>(
            perforated_surface_mesh_,
            solvers::materials::MaterialFromLameCoefficients(1, "m", approximate_mu_, approximate_lambda_));
    homogenization->Solve();
    perforated_material_ = homogenization->Coefficients();
}

auto solvers::controllers::SolverController::UniformMesh() const -> const std::shared_ptr<meshing::Mesh> & {
    return uniform_mesh_;
}
auto solvers::controllers::SolverController::PerforatedMesh() const -> const std::shared_ptr<meshing::Mesh> & {
    return perforated_mesh_;
}

void solvers::controllers::SolverController::ResetMeshPositions() {
    ResetUniformMeshPositions();
    perforated_mesh_->ResetMesh();
}

void solvers::controllers::SolverController::ResetUniformMeshPositions() { uniform_mesh_->ResetMesh(); }

void solvers::controllers::SolverController::ReloadUniformSolver(solvers::fem::LinearElastic::Type type) {
    solvers_need_reload = false;
    ResetUniformBoundaryConditions();

    uniform_solver_ = std::make_unique<solvers::fem::LinearElastic>(uniform_boundary_conditions_, uniform_material_,
                                                                    uniform_mesh_, type);

    if (type == solvers::fem::LinearElastic::Type::kDynamic) {
        NEON_LOG_INFO("Making integrator");
        uniform_integrator_ = std::make_unique<solvers::integrators::CentralDifferenceMethod>(
                dt_, mass_, uniform_solver_->K_e, uniform_solver_->U_e, uniform_solver_->F_e);
    }
}

void solvers::controllers::SolverController::ReloadPerforatedSolver(solvers::fem::LinearElastic::Type type) {
    solvers_need_reload = false;
    ResetPerforatedBoundaryConditions();

    perforated_solver_ = std::make_unique<solvers::fem::LinearElastic>(perforated_boundary_conditions_,
                                                                       perforated_material_, perforated_mesh_, type);

    if (type == solvers::fem::LinearElastic::Type::kDynamic) {
        perforated_integrator_ = std::make_unique<solvers::integrators::CentralDifferenceMethod>(
                dt_, mass_, perforated_solver_->K_e, perforated_solver_->U_e, perforated_solver_->F_e);
    }
}

void solvers::controllers::SolverController::SolveUniform(const bool dynamic) {
    NEON_ASSERT_ERROR(uniform_solver_ != nullptr, "Uniform solver has not been initialized");
    if (dynamic) {
        // Make sure the integrator has been initialized.
        NEON_ASSERT_ERROR(uniform_integrator_ != nullptr, "Uniform integrator has not been initialized.");
        uniform_integrator_->Solve(uniform_solver_->F_e, uniform_solver_->U_e);
    }

    uniform_solver_->Solve(uniform_displacements, uniform_stresses);
    uniform_mesh_->Update(uniform_displacements);
}

void solvers::controllers::SolverController::SolvePerforated(const bool dynamic) {
    NEON_ASSERT_ERROR(perforated_solver_ != nullptr, "Perforated solver has not been initialized");
    if (dynamic) {
        // Make sure the integrator has been initialized.
        NEON_ASSERT_ERROR(perforated_integrator_ != nullptr, "Perforated integrator has not been initialized.");
        perforated_integrator_->Solve(perforated_solver_->F_e, perforated_solver_->U_e);
    }
    perforated_solver_->Solve(perforated_displacements, perforated_stresses);
    perforated_mesh_->Update(perforated_displacements);
}

void solvers::controllers::SolverController::ResetUniformBoundaryConditions() {
    const Vector3r force(0, force_, 0);
    uniform_interior_nodes_.clear();
    uniform_force_nodes_.clear();
    uniform_fixed_nodes_.clear();
    uniform_boundary_conditions_.clear();
    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, uniform_mesh_, uniform_interior_nodes_,
                                 uniform_force_nodes_, uniform_fixed_nodes_);
    solvers::boundary_conditions::LoadBoundaryConditions(force, uniform_mesh_, uniform_force_nodes_,
                                                         uniform_interior_nodes_, uniform_boundary_conditions_);
}

void solvers::controllers::SolverController::ResetPerforatedBoundaryConditions() {
    const Vector3r force(0, force_, 0);
    perforated_interior_nodes_.clear();
    perforated_force_nodes_.clear();
    perforated_fixed_nodes_.clear();
    perforated_boundary_conditions_.clear();
    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, perforated_mesh_, perforated_interior_nodes_,
                                 perforated_force_nodes_, perforated_fixed_nodes_);
    solvers::boundary_conditions::LoadBoundaryConditions(force, perforated_mesh_, perforated_force_nodes_,
                                                         perforated_interior_nodes_, perforated_boundary_conditions_);
}

auto solvers::controllers::SolverController::AverageDisplacement() const -> Real {
    if (uniform_force_nodes_.size() == 0) { return 0; }

    Real sum = 0;
    for (const auto &n : uniform_force_nodes_) { sum += uniform_mesh_->positions.row(n).y(); }
    sum /= uniform_force_nodes_.size();
    return sum;
}
