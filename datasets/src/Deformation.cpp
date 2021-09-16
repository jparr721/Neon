// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <datasets/Deformation.h>
#include <memory>
#include <meshing/DofOptimizer.h>
#include <meshing/ImplicitSurfaceGenerator.h>
#include <solvers/FEM/LinearElastic.h>
#include <utilities/math/Numbers.h>

datasets::Deformation::Deformation(const std::string &path)
    : path_(path), csv_(path, {"E_x", "E_y", "E_z", "v_xy", "v_xz", "v_yz", "G_yz", "G_zx", "G_xy", "Displacement"}) {}

auto datasets::Deformation::Generate(const solvers::boundary_conditions::BoundaryConditions &boundary_conditions,
                                     const std::shared_ptr<meshing::Mesh> &mesh_, Real min_E, Real max_E, Real min_v,
                                     Real max_v, Real E_incr, Real v_incr) -> void {
    const auto mesh = std::make_shared<meshing::Mesh>(*mesh_);
    const int range_size = 40000 / 100;
    const VectorXr E_range = VectorXr::LinSpaced(range_size, 1000, 40000);
    std::vector<std::tuple<Real, Real, Real>> Ev;
    for (Real v = min_v; v <= max_v; v += v_incr) {
        const auto is_max_E = [&](Real target, Real E) -> DeformationBinarySearchReturnType {
            const auto mesh_clone = std::make_shared<meshing::Mesh>(*mesh);
            const auto static_solver = std::make_unique<solvers::fem::LinearElastic>(
                    boundary_conditions, E, v, mesh_clone, solvers::fem::LinearElastic::Type::kStatic);
            MatrixXr displacements;
            MatrixXr _;
            static_solver->Solve(displacements, _);

            // NOTE: This is error-prone for non-uniform uni-axial loads!!
            Real sum = 0.0;
            int force_applied_nodes_count = 0;
            for (const auto &bc : boundary_conditions) {
                if (bc.force.y() != 0) {
                    sum += mesh_clone->positions.row(bc.node).y();
                    ++force_applied_nodes_count;
                }
            }
            sum /= force_applied_nodes_count;

            DeformationBinarySearchReturnType ret;
            ret.displacement = sum;
            ret.target = target;
            ret.E = E;
            return ret;
        };

        const auto E_ret = utilities::algorithms::FnBinarySearch<decltype(is_max_E), VectorXr, Real>(
                is_max_E, E_range, 0, E_range.rows() - 1, 0.5);
        NEON_ASSERT_WARN(E_ret.displacement > 0,
                         "There exists no E and v combo that satisfies the displacement constraint");

        if (E_ret.displacement > 0) { Ev.emplace_back(E_ret.E, v, E_ret.displacement); }
    }

    for (const auto &c : Ev) {
        csv_ << std::vector<std::string>{std::to_string(std::get<0>(c)), std::to_string(std::get<1>(c)),
                                         std::to_string(std::get<2>(c))};
    }
}

auto datasets::Deformation::GenerateSearchSpace(const unsigned int shape, const Real force_min, const Real force_max,
                                                const Vector3r &min_E, const Vector3r &max_E, const Vector3r &min_v,
                                                const Vector3r &max_v, const Vector3r &min_G, const Vector3r &max_G,
                                                const Real E_incr, const Real v_incr, const Real G_incr) -> void {
    auto gen = std::make_unique<meshing::ImplicitSurfaceGenerator<Real>>(shape, shape, shape);

    MatrixXr V;
    MatrixXi F;
    gen->GenerateImplicitFunctionBasedMaterial(meshing::ImplicitSurfaceGenerator<Real>::kNoThickness, V, F);

    // Tetrahedralize the mesh uniformly.
    const auto mesh = std::make_shared<meshing::Mesh>(V, F, "Yzpq");

    // Assign boundary conditions, for now this is hard-coded as the uni-axial force
    std::vector<unsigned int> interior_nodes;
    std::vector<unsigned int> force_nodes;
    std::vector<unsigned int> fixed_nodes;

    // My God...
    NEON_LOG_INFO("Running! Go take a nap or something");
    for (int F = static_cast<int>(force_min); F < static_cast<int>(force_max); ++F) {
        interior_nodes.clear();
        force_nodes.clear();
        fixed_nodes.clear();

        solvers::boundary_conditions::BoundaryConditions boundary_conditions;
        meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, mesh, interior_nodes, force_nodes,
                                     fixed_nodes);
#pragma omp parallel for collapse(3)
        for (int E_x = static_cast<int>(min_E.x()); E_x < static_cast<int>(max_E.x());
             E_x += static_cast<int>(E_incr)) {
            for (int E_y = static_cast<int>(min_E.y()); E_y < static_cast<int>(max_E.y());
                 E_y += static_cast<int>(E_incr)) {
                for (int E_z = static_cast<int>(min_E.z()); E_z < static_cast<int>(max_E.z());
                     E_z += static_cast<int>(E_incr)) {
                    const Vector3r E(E_x, E_y, E_z);
                    for (Real v_xy = min_v.x(); v_xy < max_v.x(); v_xy += v_incr) {
                        for (Real v_xz = min_v.x(); v_xz < max_v.x(); v_xz += v_incr) {
                            for (Real v_yz = min_v.x(); v_yz < max_v.x(); v_yz += v_incr) {
                                const Vector3r v(v_xy, v_xz, v_yz);
                                for (Real G_yz = min_G.x(); G_yz < max_G.x(); G_yz += G_incr) {
                                    for (Real G_zx = min_G.x(); G_zx < max_G.x(); G_zx += G_incr) {
                                        for (Real G_xy = min_G.x(); G_xy < max_G.x(); G_xy += G_incr) {
                                            const Vector3r G(G_yz, G_zx, G_xy);

                                            const auto material = solvers::materials::OrthotropicMaterial(E, v, G);
                                            const auto mesh_clone = std::make_shared<meshing::Mesh>(*mesh);
                                            const auto solver = std::make_unique<solvers::fem::LinearElastic>(
                                                    boundary_conditions, material, mesh_clone,
                                                    solvers::fem::LinearElastic::Type::kStatic);

                                            MatrixXr displacements;
                                            MatrixXr _;
                                            solver->Solve(displacements, _);

                                            Real sum = 0;
                                            for (const auto &n : force_nodes) { sum += mesh->positions.row(n).y(); }
                                            sum /= force_nodes.size();

#pragma omp critical
                                            {
                                                csv_ << std::vector<std::string>{
                                                        std::to_string(E_x),  std::to_string(E_y),
                                                        std::to_string(E_z),  std::to_string(v_xy),
                                                        std::to_string(v_xz), std::to_string(v_yz),
                                                        std::to_string(G_yz), std::to_string(G_zx),
                                                        std::to_string(G_xy), std::to_string(sum),
                                                };
                                            };
                                        };
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    NEON_LOG_INFO("Phew! Generation done.");
}

bool datasets::Deformation::DeformationBinarySearchReturnType::Ok() const {
    return utilities::numbers::IsApprox(displacement, target, epsilon);
}
bool datasets::Deformation::DeformationBinarySearchReturnType::TooLarge() const { return displacement > target; }
bool datasets::Deformation::DeformationBinarySearchReturnType::TooSmall() const { return displacement < target; }
