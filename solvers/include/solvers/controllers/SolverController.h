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
#include <meshing/Mesh.h>
#include <meshing/include/meshing/implicit_surfaces/ImplicitSurfaceGenerator.h>
#include <solvers/FEM/LinearElastic.h>
#include <solvers/integrators/CentralDifferenceMethod.h>
#include <solvers/materials/Material.h>
#include <solvers/materials/OrthotropicMaterial.h>
#include <utilities/math/Tensors.h>

namespace solvers::controllers {
    class SolverController {
    public:
        static constexpr bool kUseDynamicSolver = true;
        static constexpr bool kUseStaticSolver = false;
        static constexpr unsigned int kUniformMeshID = 0;
        static constexpr unsigned int kPerforatedMeshID = 1;

        bool solvers_need_reload = true;

        MatrixXr uniform_displacements;
        MatrixXr uniform_stresses;

        MatrixXr perforated_displacements;
        MatrixXr perforated_stresses;

        explicit SolverController(int dim);

        SolverController(int dim, Real amplitude, Real thickness);

        void ReloadMeshes(int dim, Real amplitude, Real thickness);
        void HomogenizeVoidMesh();

        void ResetMeshPositions();
        void ResetUniformMeshPositions();
        void ReloadUniformSolver(solvers::fem::LinearElastic::Type type);
        void ReloadPerforatedSolver(solvers::fem::LinearElastic::Type type);

        void SolveUniform(bool dynamic);
        void SolvePerforated(bool dynamic);

        // Getters ========================================
        auto AverageDisplacement() const -> Real;
        auto UniformMesh() const -> const std::shared_ptr<meshing::Mesh> &;
        auto PerforatedMesh() const -> const std::shared_ptr<meshing::Mesh> &;

        auto Material() -> solvers::materials::OrthotropicMaterial & { return uniform_material_; }
        auto Lambda() -> Real & { return approximate_lambda_; }
        auto Mu() -> Real & { return approximate_mu_; }
        auto Dt() -> Real & { return dt_; }
        auto Mass() -> Real & { return mass_; }
        auto Force() -> Real & { return force_; }
        auto TetgenFlags() -> std::string & { return tetgen_flags; }
        auto SurfaceGeneratorAutoComputeArea() -> bool & { return auto_compute_area; }

        bool auto_compute_area = false;

        Real dt_ = 0.01;
        Real mass_ = 5;
        Real force_ = -100;

        Real approximate_lambda_ = 0;
        Real approximate_mu_ = 0;

        std::string tetgen_flags = "YzpqQ";

        Tensor3r perforated_surface_mesh_;

        solvers::materials::OrthotropicMaterial uniform_material_;
        solvers::materials::OrthotropicMaterial perforated_material_;

        // Node Arrangement
        std::vector<unsigned int> uniform_interior_nodes_;
        std::vector<unsigned int> uniform_force_nodes_;
        std::vector<unsigned int> uniform_fixed_nodes_;
        solvers::boundary_conditions::BoundaryConditions uniform_boundary_conditions_;

        std::vector<unsigned int> perforated_interior_nodes_;
        std::vector<unsigned int> perforated_force_nodes_;
        std::vector<unsigned int> perforated_fixed_nodes_;
        solvers::boundary_conditions::BoundaryConditions perforated_boundary_conditions_;
        // ===

        std::shared_ptr<meshing::Mesh> uniform_mesh_;
        std::shared_ptr<meshing::Mesh> perforated_mesh_;

        std::shared_ptr<solvers::fem::LinearElastic> uniform_solver_;
        std::shared_ptr<solvers::fem::LinearElastic> perforated_solver_;

        std::shared_ptr<solvers::integrators::CentralDifferenceMethod> uniform_integrator_;
        std::shared_ptr<solvers::integrators::CentralDifferenceMethod> perforated_integrator_;

        void ComputeUniformMesh(int dim);
        void ComputeVoidMesh(int dim, Real amplitude, Real thickness);
        void ResetUniformBoundaryConditions();
        void ResetPerforatedBoundaryConditions();
    };
}// namespace visualizer::controllers


#endif//NEON_SOLVERCONTROLLER_H
