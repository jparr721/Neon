// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_face_normals.h>
#include <meshing/Normals.h>
#include <meshing/implicit_surfaces/PeriodicGyroid.h>
#include <visualizer/Visualizer.h>

int main(int argc, char **argv) {
#ifdef NEON_HEADLESS_DISPLACEMENT
    visualizer::RveDims() = 5;
    visualizer::GenerateShape();
    visualizer::GenerateDisplacementDataset("Deformation_5x5.csv");
    return EXIT_SUCCESS;
#else
    // Igl's viewer requires vertex matrices to be doubles, fail if unset
#ifndef NEON_USE_DOUBLE
    throw std::runtime_exception("Please enable NEON_USE_DOUBLE to use igl viewer.");
#endif

    visualizer::Menu().callback_draw_custom_window = &visualizer::SimulationMenuWindow;
    visualizer::Menu().callback_draw_viewer_menu = &visualizer::GeometryMenu;
    visualizer::Viewer().plugins.push_back(&visualizer::Menu());

    visualizer::Viewer().callback_pre_draw = &visualizer::DrawCallback;
    visualizer::Controller() = std::make_shared<visualizer::controllers::SolverController>(
            visualizer::RveDims(), visualizer::Amplitude(), visualizer::Thickness());
    visualizer::Refresh();
    visualizer::Viewer().core().is_animating = false;
    visualizer::Viewer().launch();

#endif
}
