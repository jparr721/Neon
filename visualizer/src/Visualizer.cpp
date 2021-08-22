// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//
#include <visualizer/Visualizer.h>

visualizer::Visualizer::Visualizer() {
    // Add the menu plugin to the viewer so it shows up.
    viewer_.plugins.push_back(&menu_);
}

auto visualizer::Visualizer::SetMesh(const MatrixXr &V, const MatrixXi &F) -> void { viewer_.data().set_mesh(V, F); }

auto visualizer::Visualizer::Launch() -> void { viewer_.launch(); }

auto visualizer::Visualizer::AddObjectToViewer() -> void {}
