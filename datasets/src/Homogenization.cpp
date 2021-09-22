// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <datasets/Homogenization.h>
#include <meshing/DofOptimizer.h>
#include <meshing/include/meshing/implicit_surfaces/ImplicitSurfaceGenerator.h>
#include <solvers/FEM/LinearElastic.h>


auto datasets::FindOptimumDeformationParameters(const Real force, const Real target_deformation,
                                                const std::shared_ptr<meshing::Mesh> &mesh,
                                                const solvers::materials::OrthotropicMaterial &uniform_material_config)
        -> std::optional<solvers::materials::OrthotropicMaterial> {
    std::vector<unsigned int> interior_nodes;
    std::vector<unsigned int> force_nodes;
    std::vector<unsigned int> fixed_nodes;
    meshing::DofOptimizeUniaxial(meshing::Axis::Y, meshing::kMaxNodes, mesh, interior_nodes, force_nodes, fixed_nodes);

    solvers::boundary_conditions::BoundaryConditions boundary_conditions;
    solvers::boundary_conditions::LoadBoundaryConditions(Vector3r(0, force, 0), mesh, force_nodes, interior_nodes,
                                                         boundary_conditions);

    // Making a simplifying assumption about the pseudo-isotropic nature of youngs modulus.
    solvers::materials::OrthotropicMaterial material;
#pragma omp parallel for
    for (int E = 0; E < uniform_material_config.E_x; E += 100) {
        for (Real v = 0.0; v < 0.5; v += 0.1) {
            for (Real G = 0; G < 487; G += 1) {
                material = solvers::materials::OrthotropicMaterial(E, v, G);
                const auto mesh_clone = std::make_shared<meshing::Mesh>(*mesh);
                const auto solver =
                        std::make_unique<solvers::fem::LinearElastic>(boundary_conditions, material, mesh_clone);

                MatrixXr displacements;
                MatrixXr _;
                solver->Solve(displacements, _);

                Real deformation = 0;
                for (const auto &n : force_nodes) { deformation += displacements.row(n).y(); }
                deformation /= static_cast<Real>(force_nodes.size());

                if (utilities::math::IsApprox(deformation, target_deformation, 0.01)) { return material; }
            }
        }
    }

    return std::nullopt;
}

void datasets::MakeHomogenizationDataset(Real force, Real target_deformation,
                                         const solvers::materials::OrthotropicMaterial &uniform_material_config,
                                         unsigned int dim) {
    // First, generate the uniform material to ascertain the proper orthotropic parameters for the desired deformation.
    using Gen = meshing::implicit_surfaces::ImplicitSurfaceGenerator<Real>;
    auto generator = std::make_unique<Gen>(dim, dim, dim);

    MatrixXr V;
    MatrixXi F;
    generator->GenerateImplicitFunctionBasedMaterial(Gen::kNoThickness, 0, V, F);

    // Construct the mesh for our desired material
    const auto mesh = std::make_shared<meshing::Mesh>(V, F, "Yzpq");

    NEON_LOG_INFO("Construction of material is finished. Moving onto computation of optimum parameter list");
    const auto deformation_material_config_option =
            FindOptimumDeformationParameters(force, target_deformation, mesh, uniform_material_config);

    if (!deformation_material_config_option.has_value()) {
        NEON_LOG_SEPARATOR();
        NEON_LOG_ERROR("Failed to find optimum coefficients!! Exiting!!");
        NEON_LOG_SEPARATOR();
    }

    NEON_LOG_INFO(
            "Optimum material config determined, iterating homogenized search space to find the best periodic mesh");
    const auto &deformation_material_config = deformation_material_config_option.value();

    // Now, compute the best homogenization parameter list.
    generator = std::make_unique<Gen>(dim, dim, dim, Gen::Behavior::kIsotropic, Gen::Microstructure::kComposite);

    generator->GenerateImplicitFunctionBasedMaterial(1, true, V, F);
}
