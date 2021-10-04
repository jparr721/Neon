// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.

#include <datasets/Deformation.h>
#include <datasets/HomogenizationDataset.h>
#include <datasets/simulations/linear_elastic_50by50_cube.h>
#include <filesystem>
#include <visualizer/pipelines/BehaviorMatching.h>

pipelines::BehaviorMatching::BehaviorMatching() {
    if (!std::filesystem::exists("behavior_matching")) { std::filesystem::create_directory("behavior_matching"); }
    LoadBehaviorDataset();
    LoadHomogenizationDataset();
}

void pipelines::BehaviorMatching::Run() {}

void pipelines::BehaviorMatching::LoadBehaviorDataset() {
    const auto filename = paths.at(FilePathIndex::kBehaviorDataset);
    if (!std::filesystem::exists(filename)) { GenerateBehaviorDataset(filename); }
    const utilities::filesystem::extractor_fn<BehaviorDatasetEntry> row_extractor =
            [](const std::vector<std::string> &tokens, std::vector<BehaviorDatasetEntry> &rows) {
                const Vector3r E(std::stod(tokens.at(0)), std::stod(tokens.at(1)), std::stod(tokens.at(2)));
                const Vector3r v(std::stod(tokens.at(3)), std::stod(tokens.at(4)), std::stod(tokens.at(5)));
                const Vector3r G(std::stod(tokens.at(6)), std::stod(tokens.at(7)), std::stod(tokens.at(8)));
                const Real displacement = std::stod(tokens.at(9));
                rows.emplace_back(BehaviorDatasetEntry{E, v, G, displacement});
            };

    std::vector<std::string> keys;
    utilities::filesystem::ExtractCSVContent(filename, row_extractor, keys, behavior_dataset_);
    std::sort(behavior_dataset_.begin(), behavior_dataset_.end(),
              [](const BehaviorDatasetEntry &lhs, const BehaviorDatasetEntry &rhs) {
                  return lhs.displacement < rhs.displacement;
              });
}

void pipelines::BehaviorMatching::LoadHomogenizationDataset() {
    const auto filename = paths.at(FilePathIndex::kHomogenizationDataset);
    if (!std::filesystem::exists(filename)) { GenerateHomogenizationDataset(filename); }
    const utilities::filesystem::extractor_fn<HomogenizationDatasetEntry> row_extractor =
            [](const std::vector<std::string> &tokens, std::vector<HomogenizationDatasetEntry> &rows) {
                const Real thickness = std::stod(tokens.at(0));
                const Real amplitude = std::stod(tokens.at(1));
                const Real material_ratio = std::stod(tokens.at(2));

                Vector3r E(std::stod(tokens.at(3)), std::stod(tokens.at(4)), std::stod(tokens.at(5)));
                Vector3r v(std::stod(tokens.at(6)), std::stod(tokens.at(7)), std::stod(tokens.at(8)));
                Vector3r G(std::stod(tokens.at(9)), std::stod(tokens.at(10)), std::stod(tokens.at(11)));
                rows.emplace_back(HomogenizationDatasetEntry{thickness, amplitude, material_ratio, E, v, G});
            };

    std::vector<std::string> keys;
    utilities::filesystem::ExtractCSVContent(filename, row_extractor, keys, homogenization_dataset_);
    std::sort(homogenization_dataset_.begin(), homogenization_dataset_.end(),
              [](const HomogenizationDatasetEntry &lhs, const HomogenizationDatasetEntry &rhs) {
                  return lhs.E.x() < rhs.E.x() && lhs.E.y() < rhs.E.y() && lhs.E.z() < rhs.E.z();
              });
}

void pipelines::BehaviorMatching::GenerateBehaviorDataset(const std::string &filename) {
    using namespace simulations::static_files;
    NEON_LOG_WARN("Making a behavioral dataset, this could take awhile");
    const auto deformation_generator = std::make_unique<datasets::Deformation>(filename);
    deformation_generator->GenerateSearchSpace(50, kForceMin, kForceMax, kMinE, kMaxE, kMinv, kMaxv, kMinG, kMaxG,
                                               kEIncr, kvIncr, kGIncr);
}

void pipelines::BehaviorMatching::GenerateHomogenizationDataset(const std::string &filename) {
    using namespace simulations::static_files;
    NEON_LOG_WARN("Making homogenization dataset, this could take awhile");
    const auto material = solvers::materials::MaterialFromEandv(1, "", 120000, 0.3);
    datasets::MakeHomogenizationDataset(filename, kMinT, kMaxT, kMinA, kMaxA, kTIncr, kAIncr, material, 50);
}
