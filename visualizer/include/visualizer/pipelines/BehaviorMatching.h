// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//


#ifndef NEON_BEHAVIORMATCHING_H
#define NEON_BEHAVIORMATCHING_H

#include <utilities/math/LinearAlgebra.h>
#include <visualizer/pipelines/Pipeline.h>

namespace pipelines {
    /// The behavior matching pipeline takes the input dimensions and runs the FEM solver on them for the following:
    /// 1. Uniaxial
    /// 2. Torsional
    /// 3. Shear
    /// Homogenize based on the homogenization dataset and determine the optimum E and v that match that.
    class BehaviorMatching : public Pipeline {
    private:
        /// Vector-valued function of constant variables of all displacements for all combinations
        struct BehaviorDatasetEntry {
            Vector3r E;
            Vector3r v;
            Vector3r G;

            // Output: The resultant displacement
            Real displacement;
        };

        /// Homogenization of material lattices of consistent density
        struct HomogenizationDatasetEntry {
            // Sine-wave thickness and amplitude
            Real thickness;
            Real amplitude;

            /// The amount of material in the void-material volume (between 0 and 1)
            Real material_ratio;

            // Outputs: The resultant effective coefficients
            Vector3r E;
            Vector3r v;
            Vector3r G;
        };

    public:
        enum FilePathIndex {
            kBehaviorDataset = 0,
            kHomogenizationDataset = 1,
        };

        const std::vector<std::string> paths{"behavior_matching/behavior_data.csv",
                                             "behavior_matching/homogenization_data.csv"};

        explicit BehaviorMatching(const std::vector<std::string> &file_paths);
        void Run() override;

    private:
        std::vector<BehaviorDatasetEntry> behavior_dataset_;
        std::vector<HomogenizationDatasetEntry> homogenization_dataset_;

        void LoadBehaviorDataset();
        void LoadHomogenizationDataset();

        void GenerateBehaviorDataset(const std::string &filename);
        void GenerateHomogenizationDataset(const std::string &filename);
    };
}

#endif//NEON_BEHAVIORMATCHING_H
