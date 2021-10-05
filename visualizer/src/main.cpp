// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights
// reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public
// License v3. If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <igl/opengl/glfw/Viewer.h>
#include <meshing/implicit_surfaces/PeriodicGyroid.h>
#include <thread>
#include <utilities/runtime/NeonLog.h>
#include <visualizer/Visualizer.h>

int main(int argc, char **argv) {
    // Igl's viewer requires vertex matrices to be doubles, fail if unset
#ifndef NEON_USE_DOUBLE
    throw std::runtime_exception("Please enable NEON_USE_DOUBLE to use igl viewer.");
#endif
#ifdef NEON_HEADLESS_BEHAVIOR_MATCHING_GENERATOR
    NEON_LOG_INFO("Running behavior matching setup");
    solvers::BehaviorMatchingPipeline() = std::make_unique<pipelines::BehaviorMatching>();
#else
    solvers::Menu().callback_draw_custom_window = &solvers::SimulationMenuWindow;
    solvers::Menu().callback_draw_viewer_menu = &solvers::GeometryMenu;
    solvers::Viewer().plugins.push_back(&solvers::Menu());

    solvers::Viewer().callback_pre_draw = &solvers::DrawCallback;
    solvers::Controller() = std::make_shared<solvers::controllers::SolverController>(
            solvers::RveDims(), solvers::Amplitude(), solvers::Thickness());
    solvers::Refresh();
    solvers::Viewer().core().is_animating = false;
    solvers::Viewer().launch();
#endif
}
