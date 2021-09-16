// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#ifndef NEON_DEFORMATION_H
#define NEON_DEFORMATION_H

#include <meshing/Mesh.h>
#include <solvers/utilities/BoundaryCondition.h>
#include <unordered_map>
#include <utilities/algorithms/Algorithms.h>
#include <utilities/filesystem/CsvFile.h>
#include <utilities/math/LinearAlgebra.h>

namespace datasets {
    class Deformation {
        struct DeformationBinarySearchReturnType : public utilities::algorithms::FnBinarySearchAbstractReturnType {
            Real displacement = -1;
            Real target = -1;
            Real E = -1;
            auto TooLarge() const -> bool override;
            auto TooSmall() const -> bool override;
            auto Ok() const -> bool override;
        };

    public:
        explicit Deformation(const std::string &path);
        auto Generate(const solvers::boundary_conditions::BoundaryConditions &boundary_conditions,
                      const std::shared_ptr<meshing::Mesh> &mesh_, Real min_E = 1000, Real max_E = 40000,
                      Real min_v = 0.0, Real max_v = 0.5, Real E_incr = 1000, Real v_incr = 0.1) -> void;

        auto GenerateSearchSpace(unsigned int shape, Real force_min, Real force_max, const Vector3r &min_E,
                                 const Vector3r &max_E, const Vector3r &min_v, const Vector3r &max_v,
                                 const Vector3r &min_G, const Vector3r &max_G, Real E_incr, Real v_incr, Real G_incr)
                -> void;

    private:
        static constexpr Real epsilon = 0.01;

        const std::string path_;
        utilities::filesystem::CsvFile<std::string> csv_;
        std::vector<std::string> keys_;
        std::unordered_map<std::string, std::vector<Real>> rows_;
    };
}// namespace datasets

#endif//NEON_DEFORMATION_H
