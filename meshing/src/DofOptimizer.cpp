// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights
// reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public
// License v3. If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <algorithm>
#include <meshing/DofOptimizer.h>
#include <numeric>
#include <utility>

/// Quickly computes the optimal node arrangement for a given force axis
/// \param axis The axis we are applying the force to
/// \param max Whether or not to apply to the max surface ("top") or min
/// ("bottom") \param mesh The uniform_mesh nodes we are interacting with \param
/// active_nodes All free nodes sorted \param force_nodes All force-applied
/// nodes sorted \param fixed_nodes All fixed (unmoving) nodes sorted
void meshing::DofOptimizeUniaxial(const meshing::Axis axis, bool max,
                                  const std::shared_ptr<meshing::Mesh> &mesh,
                                  std::vector<unsigned int> &interior_nodes,
                                  std::vector<unsigned int> &force_nodes,
                                  std::vector<unsigned int> &fixed_nodes) {
  NEON_ASSERT_ERROR(axis != Axis::ALL,
                    "Cannot apply uniaxial force to all axes");
  // Force is applied to the surface nodes at this axis.
  FindSurfaceNodes(axis, max, mesh, force_nodes);

  // Fixed nodes are the opposite end for uniaxial.
  FindSurfaceNodes(axis, !max, mesh, fixed_nodes);

  std::vector<unsigned int> ignored;
  std::set_union(force_nodes.begin(), force_nodes.end(), fixed_nodes.begin(),
                 fixed_nodes.end(), std::back_inserter(ignored));

  // Active nodes are everything else
  FindInteriorNodes(ignored, mesh->positions.rows(), interior_nodes);
}

/// Creates a multi-axial distribution across distinct axes. This currently is
/// bug prone due to conflicting axes assigning nodes as fixed and un-fixed and
/// will be fixed later. \param axes \param min_max \param fixed_axes \param
/// min_max_fixed \param mesh \param active_nodes \param force_nodes \param
/// fixed_nodes
void meshing::DofOptimizeMultiAxial(const std::vector<Axis> &axes,
                                    const std::vector<bool> &min_max,
                                    const std::vector<Axis> &fixed_axes,
                                    const std::vector<bool> &min_max_fixed,
                                    const std::shared_ptr<meshing::Mesh> &mesh,
                                    std::vector<unsigned int> &active_nodes,
                                    std::vector<unsigned int> &force_nodes,
                                    std::vector<unsigned int> &fixed_nodes) {
  NEON_ASSERT_ERROR(axes.size() == min_max.size(),
                    "Axes and axis specification vectors must be a bijection");
  NEON_ASSERT_ERROR(!(axes.size() > 6),
                    "Cannot apply surface-wide forces in more than 2 places");

  // Find all force applied nodes
  std::vector<std::vector<unsigned int>> separated_force_nodes;
  separated_force_nodes.resize(axes.size());
  for (int i = 0; i < axes.size(); ++i) {
    FindSurfaceNodes(axes.at(i), min_max.at(i), mesh,
                     separated_force_nodes.at(i));
  }

  std::size_t separated_force_nodes_size = 0;
  for (const auto &node : separated_force_nodes) {
    separated_force_nodes_size += node.size();
  }

  force_nodes.reserve(separated_force_nodes_size);
  for (const auto &node : separated_force_nodes) {
    std::move(node.begin(), node.end(), std::back_inserter(force_nodes));
  }
  std::sort(force_nodes.begin(), force_nodes.end());
  force_nodes.erase(std::unique(force_nodes.begin(), force_nodes.end()),
                    force_nodes.end());

  // Find all fixed nodes
  std::vector<std::vector<unsigned int>> separated_fixed_nodes;
  separated_fixed_nodes.resize(axes.size());
  for (int i = 0; i < axes.size(); ++i) {
    FindSurfaceNodes(fixed_axes.at(i), min_max_fixed.at(i), mesh,
                     separated_fixed_nodes.at(i));
  }

  std::size_t separated_fixed_nodes_size = 0;
  for (const auto &node : separated_fixed_nodes) {
    separated_fixed_nodes_size += node.size();
  }

  fixed_nodes.reserve(separated_fixed_nodes_size);
  for (const auto &node : separated_fixed_nodes) {
    std::move(node.begin(), node.end(), std::back_inserter(fixed_nodes));
  }
  std::sort(fixed_nodes.begin(), fixed_nodes.end());
  fixed_nodes.erase(std::unique(fixed_nodes.begin(), fixed_nodes.end()),
                    fixed_nodes.end());

  std::vector<unsigned int> ignored;
  std::set_union(force_nodes.begin(), force_nodes.end(), fixed_nodes.begin(),
                 fixed_nodes.end(), std::back_inserter(ignored));

  // Find the rest of the nodes
  FindInteriorNodes(ignored, mesh->positions.rows(), active_nodes);
}

/// This method clears all forces from the node meshes for a given axis or all
/// \param axis The axis to apply the clear operation to
/// \param mesh The uniform_mesh we are indexing into
/// \param active_nodes The sorted list of currently active nodes
/// \param force_nodes The sorted list of current force applied nodes
/// \param fixed_nodes The sorted list of current fixed nodes
void meshing::DofOptimizeClear(meshing::Axis axis, const bool max,
                               const std::shared_ptr<meshing::Mesh> &mesh,
                               std::vector<unsigned int> &active_nodes,
                               std::vector<unsigned int> &force_nodes,
                               std::vector<unsigned int> &fixed_nodes) {
  NEON_ASSERT_ERROR(!force_nodes.empty(), "No force applied nodes to clear");

  if (axis == Axis::ALL) {
    std::vector<unsigned int> all_active_nodes;
    all_active_nodes.resize(active_nodes.size() + force_nodes.size());
    all_active_nodes.insert(all_active_nodes.begin(), force_nodes.begin(),
                            force_nodes.end());
    active_nodes = all_active_nodes;
    return;
  }

  std::vector<unsigned int> force_removed_nodes;
  FindSurfaceNodes(axis, max, mesh, force_removed_nodes);

  std::set_difference(force_nodes.begin(), force_nodes.end(),
                      force_removed_nodes.begin(), force_removed_nodes.end(),
                      active_nodes.end());
}

void meshing::FindSurfaceNodes(meshing::Axis axis, const bool max,
                               const std::shared_ptr<meshing::Mesh> &mesh,
                               std::vector<unsigned int> &nodes) {
  Real axis_coeff = max ? mesh->positions.col(axis).maxCoeff()
                        : mesh->positions.col(axis).minCoeff();
  for (int row = 0; row < mesh->positions.rows(); ++row) {
    if (mesh->positions.row(row)(axis) == axis_coeff) {
      nodes.emplace_back(row);
    }
  }

  // If our number of nodes is suspiciously small (< 10% of the mesh), we
  // initiate a re-check with some epsilon.
  nodes.clear();
  if (nodes.size() < mesh->positions.rows() * 0.1) {
    // Get the value range of the positions on this axis.
    const Real axis_distance = mesh->positions.col(axis).maxCoeff() -
                               mesh->positions.col(axis).minCoeff();

    // Take 1% of the axis distance and apply that as the epsilon.
    const Real epsilon = axis_distance * 0.01;

    for (int row = 0; row < mesh->positions.rows(); ++row) {
      const Real axis_value = mesh->positions.row(row)(axis);
      if (max) {
        if (axis_value >= (axis_coeff - epsilon)) {
          nodes.emplace_back(row);
        }
      }

      if (!max) {
        if (axis_value <= (axis_coeff + epsilon)) {
          nodes.emplace_back(row);
        }
      }
    }
  }
}

void meshing::FindSurfaceNodes(std::vector<Axis> axes, const bool max,
                               const std::shared_ptr<meshing::Mesh> &mesh,
                               std::vector<unsigned int> &nodes) {
  for (const auto &axis : axes) {
    FindSurfaceNodes(axis, max, mesh, nodes);
  }
}

/// Finds the interior nodes given a total number and a list of excluded nodes.
/// \param excluded The sorted list of excluded nodes
/// \param total_nodes The number of total nodes we are putting into the
/// interior \param interior_nodes The interior nodes which are active but have
/// no force applied
void meshing::FindInteriorNodes(const std::vector<unsigned int> &excluded,
                                std::size_t total_nodes,
                                std::vector<unsigned int> &interior_nodes) {
  interior_nodes.resize(total_nodes - excluded.size());
  unsigned int idx = 0;
  unsigned int interior_node_number = 0;
  for (unsigned int excluded_node_number : excluded) {
    if (interior_node_number == excluded_node_number) {
      ++interior_node_number;
      continue;
    } else {
      while (interior_node_number != excluded_node_number) {
        interior_nodes.at(idx) = interior_node_number;
        ++idx;
        ++interior_node_number;
      }
    }
    ++interior_node_number;
  }

  for (; idx < total_nodes - excluded.size(); ++idx, ++interior_node_number) {
    interior_nodes.at(idx) = interior_node_number;
  }
}
