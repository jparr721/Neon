// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <datasets/SolverMask.h>
#include <filesystem>
#include <meshing/ImplicitSurfaceGenerator.h>

datasets::DynamicSolverDataset::DynamicSolverDataset(const unsigned int shape, int entries) {
    input = Eigen::Tensor<Real, 5>(shape, shape, 6, shape, entries);
    target = Eigen::Tensor<Real, 5>(shape, shape, 6, shape, entries);
}

void datasets::DynamicSolverDataset::AddInputEntry(const int layer, const int feature, const MatrixXr &data) {
    const unsigned int rows = input.dimension(0);
    NEON_ASSERT_ERROR(data.rows() == rows, "Rows don't match data dimension, got: ", data.rows(), " want: ", rows);
    const unsigned int cols = input.dimension(1);
    NEON_ASSERT_ERROR(data.cols() == cols, "Cols don't match data dimension, got: ", data.cols(), " want: ", cols);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) { input(row, col, feature, layer, current_input_entry) = data(row, col); }
    }

    ++current_input_entry;
}

void datasets::DynamicSolverDataset::AddTargetEntry(const int layer, const int feature, const MatrixXr &data) {
    const unsigned int rows = target.dimension(0);
    NEON_ASSERT_ERROR(data.rows() == rows, "Rows don't match data dimension, got: ", data.rows(), " want: ", rows);
    const unsigned int cols = target.dimension(1);
    NEON_ASSERT_ERROR(data.cols() == cols, "Cols don't match data dimension, got: ", data.cols(), " want: ", cols);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            target(row, col, feature, layer, current_target_entry) = data(row, col);
        }
    }

    ++current_target_entry;
}

auto datasets::DynamicSolverDataset::Shape() const -> const unsigned int { return input.dimension(0); }
auto datasets::DynamicSolverDataset::Entries() const -> const unsigned int { return input.dimension(4); }

void datasets::DynamicSolverDataset::Sides(std::vector<MatrixXr> &bitmasks) {}
void datasets::DynamicSolverDataset::Tops(std::vector<MatrixXr> &bitmasks) {}
void datasets::DynamicSolverDataset::Fronts(std::vector<MatrixXr> &bitmasks) {}


datasets::DynamicSolverMask::DynamicSolverMask(const unsigned int shape, int entries) : dataset_(shape, entries) {
    if (std::filesystem::exists("input")) {
        NEON_LOG_WARN("Input output file was found, over-writing");
        std::filesystem::remove("input");
    }

    input_file_.open("input", std::fstream::in | std::fstream::out | std::fstream::app);

    if (std::filesystem::exists("targets")) {
        NEON_LOG_WARN("Targets output file was found, over-writing");
        std::filesystem::remove("targets");
    }

    target_file_.open("target", std::fstream::in | std::fstream::out | std::fstream::app);
}

datasets::DynamicSolverMask::~DynamicSolverMask() {
    input_file_.flush();
    input_file_.close();

    target_file_.flush();
    target_file_.close();
}

void datasets::DynamicSolverMask::GenerateDataset(const Vector3r &force, const Real mass, const Real dt, const Real E,
                                                  const Real v, const Real G) {
    const auto shape = dataset_.Shape();
    auto gen = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(shape, shape, shape);

    MatrixXr V;
    MatrixXi F;
    gen->GenerateImplicitFunctionBasedMaterial(meshing::ImplicitSurfaceGenerator<Real>::kNoThickness, V, F);

    // Tetrahedralize the mesh and do not allow superfluous tetrahedra to be added.
    mesh_ = std::make_unique<meshing::Mesh>(V, F, "Yzpq");

    std::vector<unsigned int> interior_nodes;
    std::vector<unsigned int> force_nodes;
    std::vector<unsigned int> fixed_nodes;

    // Boundary condition for uni-axial pressure on a given axis.
    solvers::boundary_conditions::BoundaryConditions boundary_conditions;

    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, mesh_, interior_nodes, force_nodes, fixed_nodes);
    solvers::boundary_conditions::LoadBoundaryConditions(force, mesh_, force_nodes, interior_nodes,
                                                         boundary_conditions);

    const auto material = solvers::materials::OrthotropicMaterial(E, v, G);

    // Solver for a dynamic-based problem
    solver_ = std::make_unique<solvers::fem::LinearElastic>(boundary_conditions, material, mesh_,
                                                            solvers::fem::LinearElastic::Type::kDynamic);

    // Integrate for dynamic problems
    integrator_ = std::make_unique<solvers::integrators::CentralDifferenceMethod>(dt, mass, solver_->K_e, solver_->U_e,
                                                                                  solver_->F_e);

    // Generate the dataset of the displacements for a given number of iterations.
    NEON_LOG_INFO(dataset_.Entries());
    for (int entry = 0; entry < dataset_.Entries(); ++entry) {
        NEON_LOG_INFO("Entry: ", entry);
        MatrixXr displacements;
        MatrixXr stresses;
        MatrixXr velocity;

        Solve(displacements, stresses);

        // Input - Mask & Stresses
        const auto mask = uniform_mesh_->ToScalarField(dim * 3);
    }
}

void datasets::DynamicSolverMask::Save() {}

void datasets::DynamicSolverMask::Solve(MatrixXr &displacements, MatrixXr &stresses) {
    NEON_ASSERT_ERROR(mesh_ != nullptr, "No mesh found");
    NEON_ASSERT_ERROR(solver_ != nullptr, "No solver found");
    NEON_ASSERT_ERROR(integrator_ != nullptr, "No integrator found");

    integrator_->Solve(solver_->F_e, solver_->U_e);
    solver_->Solve(displacements, stresses);
}

void datasets::DynamicSolverMask::VectorToOrientedMatrix(const VectorXr &v, MatrixXr &m) {
    m = (utilities::math::VectorToMatrix(v, 3, v.rows() / 3).transpose()).eval();
}

void datasets::DynamicSolverMask::SetZerosFromBoundaryConditions(
        const VectorXr &v, const solvers::boundary_conditions::BoundaryConditions &boundary_conditions, MatrixXr &m) {
    VectorXr V = VectorXr::Zero(mesh_->positions.rows() * 3);

    int i = 0;
    for (const auto &[node, _] : boundary_conditions) {
        V.segment(node * 3, 3) << v(i), v(i + 1), v(i + 2);
        i += 3;
    }

    VectorToOrientedMatrix(V, m);
}
