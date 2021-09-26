// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#define BOOST_TEST_MODULE DofOptimizerTests
#define BOOST_DYN_TEST_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <memory>
#include <meshing/DofOptimizer.h>

void P(const std::vector<unsigned int> &v) {
    for (const auto &e : v) { std::cout << e << ", "; }
    std::cout << std::endl;
}

const std::string path = "Assets/cube.ply";

BOOST_AUTO_TEST_CASE(TestDofOptimizeUniaxialMax) {
    const auto mesh = std::make_shared<meshing::Mesh>(path, "Yzpq", meshing::MeshFileType::kPly);
    const meshing::Axis axis = meshing::Axis::X;

    std::vector<unsigned int> active;
    std::vector<unsigned int> force;
    std::vector<unsigned int> fixed;

    meshing::DofOptimizeUniaxial(axis, meshing::kMaxNodes, mesh, active, force, fixed);

    const std::vector<unsigned int> active_comp{4, 5, 6, 7};
    const std::vector<unsigned int> force_comp{4, 5, 6, 7};
    const std::vector<unsigned int> fixed_comp{0, 1, 2, 3};

    BOOST_REQUIRE(active == active_comp);
    BOOST_REQUIRE(force == force_comp);
    BOOST_REQUIRE(fixed == fixed_comp);
}

BOOST_AUTO_TEST_CASE(TestDofOptimizeUniaxialMin) {
    const auto mesh = std::make_shared<meshing::Mesh>(path, "Yzpq", meshing::MeshFileType::kPly);
    const meshing::Axis axis = meshing::Axis::X;

    std::vector<unsigned int> active;
    std::vector<unsigned int> force;
    std::vector<unsigned int> fixed;

    meshing::DofOptimizeUniaxial(axis, meshing::kMinNodes, mesh, active, force, fixed);

    const std::vector<unsigned int> active_comp{0, 1, 2, 3};
    const std::vector<unsigned int> force_comp{0, 1, 2, 3};
    const std::vector<unsigned int> fixed_comp{4, 5, 6, 7};

    BOOST_REQUIRE(active == active_comp);
    BOOST_REQUIRE(force == force_comp);
    BOOST_REQUIRE(fixed == fixed_comp);
}

BOOST_AUTO_TEST_CASE(TestDofOptimizeMultiaxial) {
    const auto mesh = std::make_shared<meshing::Mesh>(path, "Yzpq", meshing::MeshFileType::kPly);
    const std::vector<meshing::Axis> axes = std::vector{meshing::Axis::X, meshing::Axis::Z};
    const std::vector<meshing::Axis> fixed_axes = std::vector{meshing::Axis::X, meshing::Axis::Z};

    std::vector<unsigned int> active;
    std::vector<unsigned int> force;
    std::vector<unsigned int> fixed;

    meshing::DofOptimizeMultiAxial(axes, std::vector<bool>{meshing::kMaxNodes, meshing::kMaxNodes}, fixed_axes,
                                   std::vector<bool>{meshing::kMinNodes, meshing::kMinNodes}, mesh, active, force,
                                   fixed);

    const std::vector<unsigned int> active_comp{5, 6};
    const std::vector<unsigned int> force_comp{1, 2, 4, 5, 6, 7};
    const std::vector<unsigned int> fixed_comp{0, 1, 2, 3, 4, 7};

    BOOST_REQUIRE(active == active_comp);
    BOOST_REQUIRE(force == force_comp);
    BOOST_REQUIRE(fixed == fixed_comp);
}