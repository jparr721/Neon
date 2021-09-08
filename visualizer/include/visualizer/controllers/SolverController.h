// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_SOLVERCONTROLLER_H
#define NEON_SOLVERCONTROLLER_H

#include <memory>
#include <meshing/ImplicitSurfaceGenerator.h>
#include <meshing/Mesh.h>
#include <solvers/FEM/LinearElastic.h>
#include <solvers/integrators/CentralDifferenceMethod.h>
#include <solvers/materials/Material.h>
#include <solvers/materials/OrthotropicMaterial.h>

namespace visualizer::controllers {
    class SolverController {
    public:
        MatrixXr displacements;
        MatrixXr stresses;

        SolverController(int dim, int void_dim, int thickness);

        void ReloadMeshes(int dim, int void_dim, int thickness);
        void HomogenizeVoidMesh(const solvers::materials::Material &material);

        void ResetMeshPositions();
        void ReloadSolvers(solvers::fem::LinearElastic::Type type);

        // Getters ========================================
        // TODO(@jparr721) Const reference
        auto UniformMesh() -> std::shared_ptr<meshing::Mesh> &;
        auto UniformIntegrator() -> std::shared_ptr<solvers::integrators::CentralDifferenceMethod> &;

        auto PerforatedMesh() -> std::shared_ptr<meshing::Mesh> &;
        auto PerforatedIntegrator() -> std::shared_ptr<solvers::integrators::CentralDifferenceMethod> &;

        auto UniformSolver() -> std::shared_ptr<solvers::fem::LinearElastic> &;
        auto PerforatedSolver() -> std::shared_ptr<solvers::fem::LinearElastic> &;

        // Setters ========================================
        void SetMaterial(const solvers::materials::OrthotropicMaterial &material) { material_ = material; }

    private:
        // TODO(@jparr721) Make this tunable.
        constexpr static Real dt_ = 0.01;

        constexpr static Real mass_ = 5;

        Real force_ = -100;
        Real E_baseline = 10000;
        Real v_baseline = 0.3;
        Real G_baseline = E_baseline / (2 * (1 + v_baseline));

        const std::string tetgen_flags = "Yzpq";

        std::vector<unsigned int> interior_nodes_;
        std::vector<unsigned int> force_nodes_;
        std::vector<unsigned int> fixed_nodes_;

        Tensor3r perforated_surface_mesh_;

        solvers::materials::OrthotropicMaterial material_;

        std::shared_ptr<meshing::Mesh> uniform_mesh_;
        std::shared_ptr<meshing::Mesh> perforated_mesh_;

        std::shared_ptr<solvers::fem::LinearElastic> uniform_solver_;
        std::shared_ptr<solvers::fem::LinearElastic> perforated_solver_;

        std::shared_ptr<solvers::integrators::CentralDifferenceMethod> uniform_integrator_;
        std::shared_ptr<solvers::integrators::CentralDifferenceMethod> perforated_integrator_;

        void ComputeUniformMesh(int dim);
        void ComputeVoidMesh(int dim, int void_dim, int thickness);

        auto ComputeActiveDofs() -> solvers::boundary_conditions::BoundaryConditions;
    };
}// namespace visualizer::controllers


#endif//NEON_SOLVERCONTROLLER_H
