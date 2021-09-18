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

datasets::DynamicSolverDataset::DynamicSolverDataset(const unsigned int shape_x, const unsigned int shape_y,
                                                     const unsigned int entries) {
    input = Tensor(shape_x, shape_y, kDatasetFeatures, entries);
    target = Tensor(shape_x, shape_y, kDatasetFeatures, entries);
}

void datasets::DynamicSolverDataset::AddInputEntry(const int feature, const MatrixXr &data) {
    if (current_input_entry == input.dimension(3)) { return; }

    const unsigned int rows = input.dimension(0);
    NEON_ASSERT_ERROR(data.rows() == rows, "Rows don't match data dimension, got: ", data.rows(), " want: ", rows);
    const unsigned int cols = input.dimension(1);
    NEON_ASSERT_ERROR(data.cols() == cols, "Cols don't match data dimension, got: ", data.cols(), " want: ", cols);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) { input(row, col, feature, current_input_entry) = data(row, col); }
    }
}

void datasets::DynamicSolverDataset::AddTargetEntry(const int feature, const MatrixXr &data) {
    if (current_target_entry == target.dimension(3)) { return; }

    const unsigned int rows = target.dimension(0);
    NEON_ASSERT_ERROR(data.rows() == rows, "Rows don't match data dimension, got: ", data.rows(), " want: ", rows);
    const unsigned int cols = target.dimension(1);
    NEON_ASSERT_ERROR(data.cols() == cols, "Cols don't match data dimension, got: ", data.cols(), " want: ", cols);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) { target(row, col, feature, current_target_entry) = data(row, col); }
    }
}

auto datasets::DynamicSolverDataset::Shape() const -> const unsigned int { return input.dimension(0); }
auto datasets::DynamicSolverDataset::Entries() const -> const unsigned int { return input.dimension(kTensorRank - 1); }

void datasets::DynamicSolverDataset::Sides(std::vector<MatrixXr> &bitmasks) {}
void datasets::DynamicSolverDataset::Tops(std::vector<MatrixXr> &bitmasks) {}
void datasets::DynamicSolverDataset::Fronts(std::vector<MatrixXr> &bitmasks) {}


datasets::DynamicSolverMask::DynamicSolverMask(const unsigned int shape, int entries)
    : mesh_shape(shape), n_entries(entries) {
    if (std::filesystem::exists(DynamicSolverDataset::kInputFileName)) {
        NEON_LOG_WARN("Input output file was found, over-writing");
        std::filesystem::remove(DynamicSolverDataset::kInputFileName);
    }

    input_file_.open(DynamicSolverDataset::kInputFileName, std::fstream::in | std::fstream::out | std::fstream::app);

    if (std::filesystem::exists(DynamicSolverDataset::kTargetFileName)) {
        NEON_LOG_WARN("Targets output file was found, over-writing");
        std::filesystem::remove(DynamicSolverDataset::kTargetFileName);
    }

    target_file_.open(DynamicSolverDataset::kTargetFileName, std::fstream::in | std::fstream::out | std::fstream::app);
}

datasets::DynamicSolverMask::~DynamicSolverMask() {
    input_file_.flush();
    input_file_.close();

    target_file_.flush();
    target_file_.close();
}

void datasets::DynamicSolverMask::GenerateDataset(const Vector3r &force, const Real mass, const Real dt, const Real E,
                                                  const Real v, const Real G) {
    //    auto gen = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(mesh_shape, mesh_shape, mesh_shape);
    //
    //    MatrixXr V;
    //    MatrixXi F;
    //    gen->GenerateImplicitFunctionBasedMaterial(meshing::ImplicitSurfaceGenerator<Real>::kNoThickness, V, F);
    //
    //    // Tetrahedralize the mesh and do not allow superfluous tetrahedra to be added.
    //    mesh_ = std::make_unique<meshing::Mesh>(V, F, "Yzpq");
    //
    //    std::vector<unsigned int> interior_nodes;
    //    std::vector<unsigned int> force_nodes;
    //    std::vector<unsigned int> fixed_nodes;
    //
    //    // Boundary condition for uni-axial pressure on a given axis.
    //    solvers::boundary_conditions::BoundaryConditions boundary_conditions;
    //
    //    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, mesh_, interior_nodes, force_nodes, fixed_nodes);
    //    solvers::boundary_conditions::LoadBoundaryConditions(force, mesh_, force_nodes, interior_nodes,
    //                                                         boundary_conditions);
    //
    //    const auto material = solvers::materials::OrthotropicMaterial(E, v, G);
    //
    //    auto dataset = DynamicSolverDataset(mesh_->positions.rows(), mesh_->positions.cols(), n_entries);
    //
    //    // Header
    //    input_file_ << dataset.input.dimension(0) << " " << dataset.input.dimension(1) << " " << dataset.input.dimension(2)
    //                << " " << dataset.input.dimension(3) << "\n";
    //    NEON_LOG_INFO("Input file setup");
    //    target_file_ << dataset.target.dimension(0) << " " << dataset.target.dimension(1) << " "
    //                 << dataset.target.dimension(2) << " " << dataset.target.dimension(3) << "\n";
    //    NEON_LOG_INFO("Target file setup");
    //
    //    // Generate the dataset of the displacements for a given number of iterations.
    //    for (Real force = 0; force < 100; ++force) {
    //        // Solver for a dynamic-based problem
    //        solver_ = std::make_unique<solvers::fem::LinearElastic>(boundary_conditions, material, mesh_,
    //                                                                solvers::fem::LinearElastic::Type::kStatic);
    //        MatrixXr scalar_field;
    //        MatrixXr forces;
    //
    //        MatrixXr displacements;
    //        MatrixXr velocity;
    //
    //        MatrixXr _;
    //        solver_->Solve(displacements, _);
    //
    //        // Average the active-axis displacement (for the scalar field)
    //        Real sum = 0;
    //        for (const auto &force_node : force_nodes) {
    //            sum += displacements.row(force_node).y();
    //        }
    //        sum /= force_nodes.size();
    //
    //        const Tensor3r tensor_field = mesh_->ToScalarField(mesh_shape * 3);
    //        const MatrixXr _sf = tensor_field.Layer(mesh_shape);
    //        PruneZeros(_sf, scalar_field);
    //
    //        NEON_LOG_INFO(scalar_field);
    //
    //        // Input - Forces & Positions
    //        SetZerosFromBoundaryConditions(solver_->F_e, boundary_conditions, forces);
    //
    //        // Target - Displacements & Velocity
    //        SetZerosFromBoundaryConditions(integrator_->Velocity(), boundary_conditions, velocity);
    //
    //        dataset.AddInputEntry(DynamicSolverDataset::kFeatureScalarField, forces);
    //        dataset.AddInputEntry(DynamicSolverDataset::kFeatureForces, forces);
    //        ++dataset.current_input_entry;
    //
    //        dataset.AddTargetEntry(DynamicSolverDataset::kDisplacements, displacements);
    //        dataset.AddTargetEntry(DynamicSolverDataset::kVelocity, velocity);
    //        ++dataset.current_target_entry;
    //
    //        mesh_->ResetMesh();
    //    }
    //
    //    // Save our outputs and we're good to go!
    //    input_file_ << dataset.input;
    //    target_file_ << dataset.target;
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

void datasets::DynamicSolverMask::PruneZeros(const MatrixXr &scalar_field, MatrixXr &pruned) {
    //    // Get the max of each row and column. For all rows/cols that are == 0, exclude.
    //    const VectorXi row_max = scalar_field.rowwise().maxCoeff();
    //    const VectorXi col_max = scalar_field.colwise().maxCoeff();
    //
    //    // Obtain the rows which are non zero and the columns. Assumes boundary sections are contiguous.
    //    unsigned int rc;
    //    VectorXi RI;
    //    utilities::math::NNZ(row_max, RI, rc, true);
    //
    //    unsigned int cc;
    //    VectorXi CI;
    //    utilities::math::NNZ(col_max, CI, cc, true);
    //
    //    pruned.resize(rc, cc);
    //    for (int row = 0; row < rc; ++row) {
    //        for (int col = 0; col < cc; ++col) {
    //            pruned(row, col) = scalar_field(RI(row), CI(col));
    //        }
    //    }
}
