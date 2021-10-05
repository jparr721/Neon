// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <datasets/Static.h>
#include <memory>
#include <meshing/Mesh.h>
#include <meshing/include/meshing/implicit_surfaces/ImplicitSurfaceGenerator.h>
#include <solvers/FEM/LinearElastic.h>
#include <vector>

void datasets::MakeUniaxialStaticSolverDataset2D(const Real force_min, const Real force_max, const unsigned int dim) {
    using Gen = meshing::implicit_surfaces::ImplicitSurfaceGenerator<Real>;
    Tensor4<Real> input_dataset(dim, dim, 1, static_cast<int>(force_max - force_min));
    Tensor4<Real> output_dataset(dim, dim, 1, static_cast<int>(force_max - force_min));

    unsigned int c_entry = 0;
#pragma omp parallel for
    for (int force = static_cast<int>(force_min); force < static_cast<int>(force_max); ++force) {
        // Generate a base scalar mesh size dimxdim
        auto generator = std::make_unique<Gen>(dim, dim, dim);

        MatrixXr V;
        MatrixXi F;
        generator->GenerateImplicitFunctionBasedMaterial(Gen::kNoThickness, 0, V, F);

        // Non-marching-cubes configured mesh.
        const Tensor3r tensor_field = generator->Surface();

        // Take the first layer off the top
        MatrixXr scalar_field = tensor_field.Layer(0);

        // Apply the force for the uni-axial strain
        MatrixXr force_field = scalar_field * -1 * force;

        // Make a mesh object for the sim calculation
        const auto mesh = std::make_shared<meshing::Mesh>(V, F, "Yzpq");

        // Dataset Generation ==============================
        // Interior
        std::vector<unsigned int> in;

        // Force
        std::vector<unsigned int> fn;

        // Fixed
        std::vector<unsigned int> xn;

        solvers::boundary_conditions::BoundaryConditions boundary_conditions;
        meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, mesh, in, fn, xn);
        solvers::boundary_conditions::LoadBoundaryConditions(Vector3r(0, -1 * force, 0), mesh, fn, in,
                                                             boundary_conditions);

        // Set a default material (for now)
        constexpr Real E = 30000;
        constexpr Real v = 0.3;
        constexpr Real G = 11538;
        const auto material = solvers::materials::OrthotropicMaterial(E, v, G);

        const auto solver = std::make_unique<solvers::fem::LinearElastic>(boundary_conditions, material, mesh,
                                                                          solvers::fem::LinearElastic::Type::kStatic);
        MatrixXr displacement;
        MatrixXr _;
        solver->Solve(displacement, _);

        // Sum the displacement value
        Real avg_U = 0;
        for (const auto &n : fn) { avg_U += displacement.row(n).y(); }
        avg_U /= static_cast<Real>(fn.size());

        MatrixXr displacement_field = scalar_field * avg_U;

        solvers::math::tensors::Assign(input_dataset, 0u, c_entry, force_field);
        solvers::math::tensors::Assign(output_dataset, 0u, c_entry, displacement_field);
        ++c_entry;
    }

    NEON_LOG_INFO("Simulation complete, saving");
    const std::string input_dataset_filename = "input_static_2d";
    solvers::math::tensors::Write(input_dataset, input_dataset_filename);

    const std::string output_dataset_filename = "output_static_2d";
    solvers::math::tensors::Write(output_dataset, output_dataset_filename);
}
