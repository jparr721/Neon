// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.

#include <datasets/Deformation.h>
#include <datasets/simulations/linear_elastic_50by50_cube.h>
#include <filesystem>
#include <utilities/runtime/NeonLog.h>
#include <visualizer/pipelines/BehaviorMatching.h>

void pipelines::BehaviorMatching::Run() {}

void pipelines::BehaviorMatching::LoadBehaviorDataset() {
    const auto filename = file_paths_.at(FilePathIndex::kBehaviorDataset);
    if (!std::filesystem::exists(filename)) { MakeBehaviorDataset(filename); }
}

void pipelines::BehaviorMatching::LoadHomogenizationDataset() {
    const auto filename = file_paths_.at(FilePathIndex::kHomogenizationDataset);
    if (!std::filesystem::exists(filename)) { MakeHomogenizationDataset(filename); }
}

void pipelines::BehaviorMatching::MakeBehaviorDataset(const std::string &filename) {
    using namespace simulations::static_files;
    NEON_LOG_WARN("Making a behavioral dataset, this could take awhile");
    const auto deformation_generator = std::make_unique<datasets::Deformation>(filename);
    deformation_generator->GenerateSearchSpace(50, kForceMin, kForceMax, kMinE, kMaxE, kMinv, kMaxv, kMinG, kMaxG,
                                               kEIncr, kvIncr, kGIncr);
}

void pipelines::BehaviorMatching::MakeHomogenizationDataset(const std::string &filename) {}