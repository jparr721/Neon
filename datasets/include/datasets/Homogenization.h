// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_HOMOGENIZATION_H
#define NEON_HOMOGENIZATION_H

#include <memory>
#include <meshing/Mesh.h>
#include <optional>
#include <solvers/materials/OrthotropicMaterial.h>
#include <utilities/math/LinearAlgebra.h>

namespace datasets {
    /// This function computes the optimum deformation with a few key simplifying assumptions:
    /// 1. We want to consider this system as a uniform cube which is non-void
    /// 2. We want to take the displacement threshold as a function of the average of the total force-applied nodes
    ///    displacements as a result of the infinitesimal strain FEM.
    /// \param force The y-axis force to apply to the nodes
    /// \param target_deformation The amount of deformation we want to optimize for.
    /// \param mesh The mesh we are rendering and operating on.
    /// \param uniform_material_config The material composition of the uniform material.
    /// \return OrthotropicMaterial the orthotropic material specification for the optimum deformation
    auto FindOptimumDeformationParameters(Real force, Real target_deformation,
                                          const std::shared_ptr<meshing::Mesh> &mesh,
                                          const solvers::materials::OrthotropicMaterial &uniform_material_config)
            -> std::optional<solvers::materials::OrthotropicMaterial>;
    void MakeHomogenizationDataset(Real force, Real target_deformation,
                                   const solvers::materials::OrthotropicMaterial &uniform_material_config,
                                   unsigned int dim);
}// namespace datasets

#endif//NEON_HOMOGENIZATION_H
