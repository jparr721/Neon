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
#include <solvers/FEM/LinearElastic.h>
#include <utilities/math/Numbers.h>

datasets::Deformation::Deformation(const std::string &path) : path_(path), csv_(path, {"E", "v", "Displacement"}) {}

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
            static_solver->SolveStatic();

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
auto datasets::Deformation::GenerateSearchSpace(
        const solvers::boundary_conditions::BoundaryConditions &boundary_conditions,
        const std::shared_ptr<meshing::Mesh> &mesh, Real min_E, Real max_E, Real min_v, Real max_v, Real E_incr,
        Real v_incr) -> void {
#pragma omp parallel for
    for (int E = static_cast<int>(min_E); E < static_cast<int>(max_E); E += static_cast<int>(E_incr)) {
        for (Real v = min_v; v < max_v; v += v_incr) {
            const auto mesh_clone = std::make_shared<meshing::Mesh>(*mesh);
            const auto static_solver = std::make_unique<solvers::fem::LinearElastic>(
                    boundary_conditions, E, v, mesh_clone, solvers::fem::LinearElastic::Type::kStatic);
            static_solver->SolveStatic();

            Real sum = 0;
            int force_applied_nodes_count = 0;
            for (const auto &bc : boundary_conditions) {
                if (bc.force.y() != 0) {
                    sum += mesh_clone->positions.row(bc.node).y();
                    ++force_applied_nodes_count;
                }
            }
            sum /= force_applied_nodes_count;

#pragma omp critical
            { csv_ << std::vector<std::string>{std::to_string(E), std::to_string(v), std::to_string(sum)}; };
        }
    }
}

bool datasets::Deformation::DeformationBinarySearchReturnType::Ok() const {
    return utilities::numbers::IsApprox(displacement, target, epsilon);
}
bool datasets::Deformation::DeformationBinarySearchReturnType::TooLarge() const { return displacement > target; }
bool datasets::Deformation::DeformationBinarySearchReturnType::TooSmall() const { return displacement < target; }
