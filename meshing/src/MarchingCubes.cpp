// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public License v3.
// If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#include <map>
#include <meshing/MarchingCubes.h>

void meshing::MarchingCubes::GenerateGeometry(unsigned int n_cells_x, unsigned int n_cells_y, unsigned int n_cells_z,
                                              MatrixXr &V, MatrixX<int> &F) {
    std::vector<Vector3<int>> triangles;
    std::map<unsigned int, Position> vertices;

    n_cells_x_ = n_cells_x;
    n_cells_y_ = n_cells_y;
    n_cells_z_ = n_cells_z;
    const unsigned int points_in_x_dir = n_cells_x_ + 1;
    const unsigned int points_in_slice = points_in_x_dir * (n_cells_y_ + 1);

    const auto insert_at_edge = [&](const unsigned int x, const unsigned int y, const unsigned int z,
                                    const unsigned int edge_number) -> void {
        const Vector3r point = Intersection(x, y, z, edge_number);
        const unsigned int id = EdgeId(x, y, z, edge_number);
        vertices.insert({id, Position{0, point}});
    };

    for (unsigned int z = 0; z < n_cells_z_; ++z) {
        for (unsigned int y = 0; y < n_cells_y_; ++y) {
            for (unsigned int x = 0; x < n_cells_x_; ++x) {
                // Calculate lookup index in table
                unsigned int table_index = 0;
                if (scalar_field_[z * points_in_slice + y * points_in_x_dir + x] < iso_level_) { table_index |= 1; }

                if (scalar_field_[z * points_in_slice + (y + 1) * points_in_x_dir + x] < iso_level_) {
                    table_index |= 2;
                }

                if (scalar_field_[z * points_in_slice + (y + 1) * points_in_x_dir + (x + 1)] < iso_level_) {
                    table_index |= 4;
                }

                if (scalar_field_[z * points_in_slice + y * points_in_x_dir + (x + 1)] < iso_level_) {
                    table_index |= 8;
                }

                if (scalar_field_[(z + 1) * points_in_slice + y * points_in_x_dir + x] < iso_level_) {
                    table_index |= 16;
                }

                if (scalar_field_[(z + 1) * points_in_slice + (y + 1) * points_in_x_dir + x] < iso_level_) {
                    table_index |= 32;
                }

                if (scalar_field_[(z + 1) * points_in_slice + (y + 1) * points_in_x_dir + (x + 1)] < iso_level_) {
                    table_index |= 64;
                }

                if (scalar_field_[(z + 1) * points_in_slice + y * points_in_x_dir + (x + 1)] < iso_level_) {
                    table_index |= 128;
                }

                // Triangulate in this cell
                if (edge_table.at(table_index) != 0) {
                    if (edge_table.at(table_index) & 8) {
                        constexpr unsigned int edge_number = 3;
                        insert_at_edge(x, y, z, edge_number);
                    }

                    if (edge_table.at(table_index) & 1) {
                        constexpr unsigned int edge_number = 0;
                        insert_at_edge(x, y, z, edge_number);
                    }

                    if (edge_table.at(table_index) & 256) {
                        constexpr unsigned int edge_number = 8;
                        insert_at_edge(x, y, z, edge_number);
                    }

                    if (x == n_cells_x_ - 1) {
                        if (edge_table.at(table_index) & 4) {
                            constexpr unsigned int edge_number = 2;
                            insert_at_edge(x, y, z, edge_number);
                        }

                        if (edge_table.at(table_index) & 2048) {
                            constexpr unsigned int edge_number = 11;
                            insert_at_edge(x, y, z, edge_number);
                        }
                    }

                    if (y == n_cells_y_ - 1) {
                        if (edge_table.at(table_index) & 2) {
                            constexpr unsigned int edge_number = 1;
                            insert_at_edge(x, y, z, edge_number);
                        }

                        if (edge_table.at(table_index) & 512) {
                            constexpr unsigned int edge_number = 9;
                            insert_at_edge(x, y, z, edge_number);
                        }
                    }

                    if (z == n_cells_z_ - 1) {
                        if (edge_table.at(table_index) & 16) {
                            constexpr unsigned int edge_number = 4;
                            insert_at_edge(x, y, z, edge_number);
                        }

                        if (edge_table.at(table_index) & 128) {
                            constexpr unsigned int edge_number = 7;
                            insert_at_edge(x, y, z, edge_number);
                        }
                    }

                    if ((x == n_cells_x_ - 1) && (y == n_cells_y_ - 1)) {
                        if (edge_table.at(table_index) & 1024) {
                            constexpr unsigned int edge_number = 10;
                            insert_at_edge(x, y, z, edge_number);
                        }
                    }

                    if ((x == n_cells_x_ - 1) && (z == n_cells_z_ - 1)) {
                        if (edge_table.at(table_index) & 64) {
                            constexpr unsigned int edge_number = 6;
                            insert_at_edge(x, y, z, edge_number);
                        }
                    }

                    if ((y == n_cells_y_ - 1) && (z == n_cells_z_ - 1)) {
                        if (edge_table.at(table_index) & 32) {
                            constexpr unsigned int edge_number = 5;
                            insert_at_edge(x, y, z, edge_number);
                        }
                    }

                    for (unsigned int i = 0; triangle_table.at(table_index).at(i) != -1; i += 3) {
                        Vector3<int> triangle;
                        triangle.x() = EdgeId(x, y, z, triangle_table.at(table_index).at(i));
                        triangle.y() = EdgeId(x, y, z, triangle_table.at(table_index).at(i + 1));
                        triangle.z() = EdgeId(x, y, z, triangle_table.at(table_index).at(i + 2));
                        triangles.emplace_back(triangle);
                    }
                }
            }
        }
    }

    assert(!triangles.empty() && "NO RECOREDED TRIANGLES");
    assert(!vertices.empty() && "NO RECORDED VERTICES");
    CalculateFacesAndVertices(V, F, triangles, vertices);
}

auto meshing::MarchingCubes::Interpolate(const Real x1, const Real y1, const Real z1, const Real x2, const Real y2,
                                         const Real z2, const Real val1, const Real val2) const -> Vector3r {
    const auto mu_interpolate = [](const Real v1, const Real v2, const Real mu) -> Real { return v1 + mu * (v2 - v1); };

    const Real mu = (iso_level_ - val1) / (val2 - val1);
    return Vector3r(mu_interpolate(x1, x2, mu), mu_interpolate(y1, y2, mu), mu_interpolate(z1, z2, mu));
}

auto meshing::MarchingCubes::Intersection(const unsigned int x, const unsigned int y, const unsigned int z,
                                          const unsigned int edge_number) -> Vector3r {
    unsigned int v1x = x;
    unsigned int v1y = y;
    unsigned int v1z = z;
    unsigned int v2x = x;
    unsigned int v2y = y;
    unsigned int v2z = z;

    switch (edge_number) {
        case 0:
            v2y += 1;
            break;
        case 1:
            v1y += 1;
            v2x += 1;
            v2y += 1;
            break;
        case 2:
            v1x += 1;
            v1y += 1;
            v2x += 1;
            break;
        case 3:
            v1x += 1;
            break;
        case 4:
            v1z += 1;
            v2y += 1;
            v2z += 1;
            break;
        case 5:
            v1y += 1;
            v1z += 1;
            v2x += 1;
            v2y += 1;
            v2z += 1;
            break;
        case 6:
            v1x += 1;
            v1y += 1;
            v1z += 1;
            v2x += 1;
            v2z += 1;
            break;
        case 7:
            v1x += 1;
            v1z += 1;
            v2z += 1;
            break;
        case 8:
            v2z += 1;
            break;
        case 9:
            v1y += 1;
            v2y += 1;
            v2z += 1;
            break;
        case 10:
            v1x += 1;
            v1y += 1;
            v2x += 1;
            v2y += 1;
            v2z += 1;
            break;
        case 11:
            v1x += 1;
            v2x += 1;
            v2z += 1;
            break;
        default:
            assert(edge_number <= 11 && edge_number >= 0 && "INVALID EDGE NUMBER FOUND");
    }

    const Real x1 = v1x * cell_length_;
    const Real y1 = v1y * cell_length_;
    const Real z1 = v1z * cell_length_;
    const Real x2 = v2x * cell_length_;
    const Real y2 = v2y * cell_length_;
    const Real z2 = v2z * cell_length_;

    const unsigned int points_in_x_direction = n_cells_x_ + 1;
    const unsigned int points_in_slice = points_in_x_direction * (n_cells_y_ + 1);
    const Real val1 = scalar_field_[v1z * points_in_slice + v1y * points_in_x_direction + v1x];
    const Real val2 = scalar_field_[v2z * points_in_slice + v2y * points_in_x_direction + v2x];
    return Interpolate(x1, y1, z1, x2, y2, z2, val1, val2);
}

auto meshing::MarchingCubes::VertexId(const unsigned int x, const unsigned int y, const unsigned int z)
        -> unsigned int {
    return 3 * (z * (n_cells_y_ + 1) * (n_cells_x_ + 1) + y * (n_cells_x_ + 1) + x);
}

auto meshing::MarchingCubes::EdgeId(const unsigned int x, const unsigned int y, const unsigned int z,
                                    const unsigned int edge_number) -> unsigned int {
    switch (edge_number) {
        case 0:
            return VertexId(x, y, z) + 1;
        case 1:
            return VertexId(x, y + 1, z);
        case 2:
            return VertexId(x + 1, y, z) + 1;
        case 3:
            return VertexId(x, y, z);
        case 4:
            return VertexId(x, y, z + 1) + 1;
        case 5:
            return VertexId(x, y + 1, z + 1);
        case 6:
            return VertexId(x + 1, y, z + 1) + 1;
        case 7:
            return VertexId(x, y, z + 1);
        case 8:
            return VertexId(x, y, z) + 2;
        case 9:
            return VertexId(x, y + 1, z) + 2;
        case 10:
            return VertexId(x + 1, y + 1, z) + 2;
        case 11:
            return VertexId(x + 1, y, z) + 2;
        default:
            return -1;
    }
}

void meshing::MarchingCubes::CalculateFacesAndVertices(MatrixXr &V, MatrixX<int> &F,
                                                       std::vector<Vector3<int>> &triangles,
                                                       std::map<unsigned int, Position> &vertices) {
    unsigned int index = 0;
    for (auto &[_, position] : vertices) {
        position.index = index;
        ++index;
    }

    for (Vector3<int> &triangle : triangles) {
        for (int i = 0; i < 3; ++i) {
            const unsigned int new_index = vertices.at(triangle(i)).index;
            triangle(i) = new_index;
        }
    }

    V.resize(vertices.size(), 3);
    unsigned int i = 0;
    for (const auto &[_, position] : vertices) {
        V.row(i) = position.position;
        ++i;
    }

    F.resize(triangles.size(), 3);
    i = 0;
    for (const auto &triangle : triangles) {
        F.row(i) = triangle;
        ++i;
    }
}
