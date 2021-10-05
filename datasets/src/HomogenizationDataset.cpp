// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <datasets/HomogenizationDataset.h>
#include <meshing/DofOptimizer.h>
#include <meshing/implicit_surfaces/PeriodicGyroid.h>
#include <solvers/FEM/LinearElastic.h>
#include <solvers/materials/ScalarField.h>
#include <utilities/filesystem/CsvFile.h>


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

                if (solvers::math::IsApprox(deformation, target_deformation, 0.01)) { return material; }
            }
        }
    }

    return std::nullopt;
}

void datasets::MakeHomogenizationDataset(const std::string &filename, Real min_thickness, Real max_thickness,
                                         Real min_amplitude, Real max_amplitude, Real thickness_incr,
                                         Real amplitude_incr, const solvers::materials::Material &material,
                                         unsigned int dim) {
    solvers::filesystem::CsvFile<std::string> csv_(filename, {"Dimensions", "Thickness", "Amplitude", "Material Ratio",
                                                              "E1", "E2", "E3", "v1", "v2", "v3", "G1", "G2", "G3"});


    for (Real t = min_thickness; t < max_thickness; t += thickness_incr) {
        for (Real a = min_amplitude; a < max_amplitude; a += amplitude_incr) {
            MatrixXr V;
            MatrixXi F;
            Tensor3r scalar_field;
            meshing::implicit_surfaces::ComputeImplicitGyroidMarchingCubes(
                    a, t, dim, meshing::implicit_surfaces::SineFunction, V, F, scalar_field);
            const auto homo = std::make_unique<solvers::materials::Homogenization>(scalar_field, material);
            homo->Solve();

            std::map<Real, Real> ratios;
            solvers::materials::scalar_field::ComputeMaterialRatio(scalar_field, ratios);

            constexpr int kMaterialValue = 1;
            const auto material_ratio = ratios[kMaterialValue];

            csv_ << std::vector<std::string>{std::to_string(dim),
                                             std::to_string(t),
                                             std::to_string(a),
                                             std::to_string(material_ratio),
                                             std::to_string(homo->Coefficients().E_x),
                                             std::to_string(homo->Coefficients().E_y),
                                             std::to_string(homo->Coefficients().E_z),
                                             std::to_string(homo->Coefficients().v_yx),
                                             std::to_string(homo->Coefficients().v_zx),
                                             std::to_string(homo->Coefficients().v_zy),
                                             std::to_string(homo->Coefficients().G_yz),
                                             std::to_string(homo->Coefficients().G_zx),
                                             std::to_string(homo->Coefficients().G_xy)};
        }
    }
}