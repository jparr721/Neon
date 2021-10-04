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
#include <meshing/include/meshing/implicit_surfaces/ImplicitSurfaceGenerator.h>
#include <solvers/FEM/LinearElastic.h>
#include <solvers/materials/Homogenization.h>
#include <utilities/math/Numbers.h>

datasets::Deformation::Deformation(const std::string &path)
    : path_(path), csv_(path, {"E_x", "E_y", "E_z", "v_xy", "v_xz", "v_yz", "G_yz", "G_zx", "G_xy", "Displacement"}) {}

void datasets::Deformation::Generate(const solvers::boundary_conditions::BoundaryConditions &boundary_conditions,
                                     const std::shared_ptr<meshing::Mesh> &mesh_, Real min_E, Real max_E, Real min_v,
                                     Real max_v, Real E_incr, Real v_incr) {
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

void datasets::Deformation::GenerateSearchSpace(unsigned int shape, const int force_min, const int force_max,
                                                const int min_E, const int max_E, const Real &min_v, const Real &max_v,
                                                const Real &min_G, const Real &max_G, int E_incr, Real v_incr,
                                                Real G_incr) {
    auto gen = std::make_unique<meshing::implicit_surfaces::ImplicitSurfaceGenerator<Real>>(shape, shape, shape);

    MatrixXr V;
    MatrixXi F;
    gen->GenerateImplicitFunctionBasedMaterial(meshing::implicit_surfaces::ImplicitSurfaceGenerator<Real>::kNoThickness,
                                               0, V, F);

    // Tetrahedralize the mesh uniformly.
    const auto mesh = std::make_shared<meshing::Mesh>(V, F, "Yzpq");

    // Assign boundary conditions, for now this is hard-coded as the uni-axial force
    std::vector<unsigned int> interior_nodes;
    std::vector<unsigned int> force_nodes;
    std::vector<unsigned int> fixed_nodes;

    // My God...
    NEON_LOG_INFO("Running! Go take a nap or something");
    for (int force = force_min; force <= force_max; ++force) {
        NEON_LOG_INFO("Iteration: ", force, " out of ", force_max);
        interior_nodes.clear();
        force_nodes.clear();
        fixed_nodes.clear();

        solvers::boundary_conditions::BoundaryConditions boundary_conditions;
        meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, mesh, interior_nodes, force_nodes,
                                     fixed_nodes);
        for (int E_x = min_E; E_x < max_E; E_x += E_incr) {
            for (int E_y = min_E; E_y < max_E; E_y += E_incr) {
                for (int E_z = min_E; E_z < max_E; E_z += E_incr) {
                    const Vector3r E(E_x, E_y, E_z);
                    for (Real v_xy = min_v; v_xy < max_v; v_xy += v_incr) {
                        for (Real v_xz = min_v; v_xz < max_v; v_xz += v_incr) {
                            for (Real v_yz = min_v; v_yz < max_v; v_yz += v_incr) {
                                const Vector3r v(v_xy, v_xz, v_yz);
                                for (Real G_yz = min_G; G_yz < max_G; G_yz += G_incr) {
                                    for (Real G_zx = min_G; G_zx < max_G; G_zx += G_incr) {
                                        for (Real G_xy = min_G; G_xy < max_G; G_xy += G_incr) {
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

                                            csv_ << std::vector<std::string>{
                                                    std::to_string(E_x),  std::to_string(E_y),  std::to_string(E_z),
                                                    std::to_string(v_xy), std::to_string(v_xz), std::to_string(v_yz),
                                                    std::to_string(G_yz), std::to_string(G_zx), std::to_string(G_xy),
                                                    std::to_string(sum),
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
    }

    NEON_LOG_INFO("Phew! Generation done.");
}

auto datasets::Deformation::DeformationBinarySearchReturnType::Ok() const -> bool {
    return utilities::numbers::IsApprox(displacement, target, epsilon);
}
auto datasets::Deformation::DeformationBinarySearchReturnType::TooLarge() const -> bool {
    return displacement > target;
}
auto datasets::Deformation::DeformationBinarySearchReturnType::TooSmall() const -> bool {
    return displacement < target;
}
