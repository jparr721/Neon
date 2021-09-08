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

        SolverController() = default;

        void ReloadMeshes(int dim, int void_dim, int thickness);
        void HomogenizeVoidMesh(const solvers::materials::Material &material);


        auto UniformSolver() const -> const std::shared_ptr<solvers::fem::LinearElastic> &;
        auto PerforatedSolver() const -> const std::shared_ptr<solvers::fem::LinearElastic> &;

        void SetMaterial(const solvers::materials::OrthotropicMaterial &material) { material_ = material; }

    private:
        const std::string tetgen_flags = "Yzpq";

        Tensor3r perforated_surface_mesh_;

        solvers::materials::OrthotropicMaterial material_;

        std::shared_ptr<meshing::Mesh> uniform_mesh_;
        std::shared_ptr<meshing::Mesh> perforated_mesh_;

        std::shared_ptr<solvers::fem::LinearElastic> uniform_solver_;
        std::shared_ptr<solvers::fem::LinearElastic> perforated_solver_;

        std::shared_ptr<solvers::integrators::CentralDifferenceMethod> uniform_integrator_;

        void ComputeUniformMesh(int dim);
        void ComputeVoidMesh(int dim, int void_dim, int thickness);
    };
}// namespace visualizer::controllers


#endif//NEON_SOLVERCONTROLLER_H
