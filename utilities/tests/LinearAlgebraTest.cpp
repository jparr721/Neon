// This file is part of Neon, an FEM research application.
//
// Copyright (c) 2021 Jarred Parr <jarred.parr@ucdenver.edu>. All rights
// reserved.
//
// This Source Code Form is subject to the terms of the GNU General Public
// License v3. If a copy of the GPL was not included with this file, you can
// obtain one at https://www.gnu.org/licenses/gpl-3.0.en.html.
//

#define BOOST_TEST_MODULE LinearAlgebraTests
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <memory>

#include <boost/test/unit_test.hpp>
#include <utilities/math/LinearAlgebra.h>
#include <utilities/math/Tensors.h>

using namespace solvers::math;

BOOST_AUTO_TEST_CASE(TestTensor3Constructor) {
  const Tensor3i t(3, 2, 3);
  BOOST_REQUIRE(t.Dimensions().size() == 3);
  BOOST_REQUIRE(t.Dimension(0) == 3);
  BOOST_REQUIRE(t.Dimension(1) == 2);
  BOOST_REQUIRE(t.Dimension(2) == 3);
}

BOOST_AUTO_TEST_CASE(TestTensor3Operator) {
  Tensor3i t(2, 2, 1);
  t.SetConstant(0);
  t(0, 0, 0) = 1;
  BOOST_REQUIRE(t(0, 0, 0) == 1);
}

BOOST_AUTO_TEST_CASE(TestTensor3SetCol) {
  Tensor3i t(2, 2, 1);
  t.SetConstant(1);

  VectorXi v(2);
  v << 5, 4;
  t.SetCol(0, 0, v);

  MatrixXi comp(2, 2);
  comp << 5, 1, 4, 1;

  BOOST_REQUIRE(t.Layer(0).isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestTensor3AppendCol) {
  Tensor3i t(2, 2, 1);
  t.SetConstant(1);

  VectorX<int> c1(2);
  c1 << 2, 3;

  std::vector<VectorX<int>> cols{c1};

  Tensor3i t2 = t.Append(cols, Tensor3i::InsertOpIndex::kStart,
                         Tensor3i::OpOrientation::kCol);
  BOOST_REQUIRE(t2.Dimension(0) == 2);
  BOOST_REQUIRE(t2.Dimension(1) == 3);
  BOOST_REQUIRE(t2.Dimension(2) == 1);

  MatrixX<int> layer_comp(2, 3);
  layer_comp.row(0) << 2, 1, 1;
  layer_comp.row(1) << 3, 1, 1;

  const MatrixX<int> m = t2.Matrix();

  BOOST_REQUIRE(m.isApprox(layer_comp));
}

BOOST_AUTO_TEST_CASE(TestTensor3AppendRow) {
  Tensor3i t(2, 2, 1);
  t.SetConstant(1);

  VectorX<int> c1(2);
  c1 << 2, 3;

  std::vector<VectorX<int>> cols{c1};

  Tensor3i t2 = t.Append(cols, Tensor3i::InsertOpIndex::kStart,
                         Tensor3i::OpOrientation::kRow);
  BOOST_REQUIRE(t2.Dimension(0) == 3);
  BOOST_REQUIRE(t2.Dimension(1) == 2);
  BOOST_REQUIRE(t2.Dimension(2) == 1);

  MatrixX<int> layer_comp(3, 2);
  layer_comp.row(0) << 2, 3;
  layer_comp.row(1) << 1, 1;
  layer_comp.row(2) << 1, 1;

  const MatrixX<int> m = t2.Matrix();

  BOOST_REQUIRE(m.isApprox(layer_comp));
}

BOOST_AUTO_TEST_CASE(TestTensor3AppendLayer) {
  Tensor3i t(2, 2, 1);
  t.SetConstant(1);

  MatrixX<int> layer(2, 2);
  layer.setConstant(2);

  Tensor3i t2 = t.Append(layer, Tensor3i::InsertOpIndex::kEnd);
  BOOST_REQUIRE(t2.Dimension(0) == 2);
  BOOST_REQUIRE(t2.Dimension(1) == 2);
  BOOST_REQUIRE(t2.Dimension(2) == 2);

  MatrixX<int> l0_comp(2, 2);
  l0_comp.setConstant(1);
  MatrixX<int> layer_0 = t2.At(0);

  BOOST_REQUIRE(layer_0.isApprox(l0_comp));

  MatrixX<int> l1_comp(2, 2);
  l1_comp.setConstant(2);
  MatrixX<int> layer_1 = t2.At(1);
  BOOST_REQUIRE(layer_1.isApprox(l1_comp));
}

BOOST_AUTO_TEST_CASE(TestTensor3AppendLayerToFront) {
  Tensor3i t(2, 2, 1);
  t.SetConstant(1);

  MatrixX<int> layer(2, 2);
  layer.setConstant(2);

  Tensor3i t2 = t.Append(layer, Tensor3i::InsertOpIndex::kStart);
  BOOST_REQUIRE(t2.Dimension(0) == 2);
  BOOST_REQUIRE(t2.Dimension(1) == 2);
  BOOST_REQUIRE(t2.Dimension(2) == 2);

  MatrixX<int> l0_comp(2, 2);
  l0_comp.setConstant(2);
  MatrixX<int> layer_0 = t2.At(0);

  BOOST_REQUIRE(layer_0.isApprox(l0_comp));

  MatrixX<int> l1_comp(2, 2);
  l1_comp.setConstant(1);
  MatrixX<int> layer_1 = t2.At(1);
  BOOST_REQUIRE(layer_1.isApprox(l1_comp));
}

BOOST_AUTO_TEST_CASE(TestTop) {
  Tensor3i t(2, 2, 2);
  t.SetConstant(1);

  MatrixXi d(2, 2);
  d.setConstant(10);
  t.SetTop(0, d);

  const MatrixXi v = t.Top(0);

  BOOST_REQUIRE(v.isApprox(d));
}

BOOST_AUTO_TEST_CASE(TestSetTop) {
  Tensor3i t(2, 2, 2);
  t.SetConstant(1);

  MatrixXi d(2, 2);
  d.setConstant(10);
  t.SetTop(0, d);

  const MatrixXi v = t.Top(0);

  BOOST_REQUIRE(v.isApprox(d));
}

BOOST_AUTO_TEST_CASE(TestSide) {
  Tensor3i t(2, 2, 2);
  t.SetConstant(1);

  MatrixXi d(2, 2);
  d.setConstant(10);
  t.SetSide(0, d);

  const MatrixXi v = t.Side(0);

  BOOST_REQUIRE(v.isApprox(d));
}

BOOST_AUTO_TEST_CASE(TestSetSide) {
  Tensor3i t(2, 2, 2);
  t.SetConstant(1);

  MatrixXi d(2, 2);
  d.setConstant(10);
  t.SetSide(0, d);

  const MatrixXi v = t.Side(0);

  BOOST_REQUIRE(v.isApprox(d));
}

BOOST_AUTO_TEST_CASE(TestIntoVector) {
  Tensor3i t(2, 2, 1);
  t.SetConstant(2);

  const VectorX<int> v = t.Vector();
  BOOST_REQUIRE(v.rows() == 4);
}

BOOST_AUTO_TEST_CASE(TestHStack) {
  MatrixX<int> one(2, 2);
  one.setConstant(1);

  MatrixX<int> two(2, 2);
  two.setConstant(2);

  MatrixX<int> three(2, 2);
  three.setConstant(3);

  MatrixX<int> comp(6, 2);
  comp << 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3;

  const MatrixX<int> stacked = solvers::math::HStack(std::vector{one, two, three});

  BOOST_REQUIRE(stacked.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestHStackVectors) {
  VectorX<int> one(2);
  one.setConstant(1);

  VectorX<int> two(2);
  two.setConstant(2);

  VectorX<int> three(2);
  three.setConstant(3);

  MatrixX<int> comp(3, 2);
  comp << 1, 1, 2, 2, 3, 3;

  const MatrixX<int> stacked = solvers::math::HStack(std::vector{one, two, three});

  BOOST_REQUIRE(stacked.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestSetLayer) {
  Tensor3i t(2, 2, 1);
  t.SetConstant(1);
  MatrixX<int> layer(2, 2);
  layer.setConstant(5);

  BOOST_REQUIRE(t(0, 0, 0) = 1);
  BOOST_REQUIRE(t(1, 0, 0) = 1);
  BOOST_REQUIRE(t(0, 1, 0) = 1);
  BOOST_REQUIRE(t(1, 1, 0) = 1);

  t.SetLayer(0, layer);
  BOOST_REQUIRE(t(0, 0, 0) = 5);
  BOOST_REQUIRE(t(1, 0, 0) = 5);
  BOOST_REQUIRE(t(0, 1, 0) = 5);
  BOOST_REQUIRE(t(1, 1, 0) = 5);
}

BOOST_AUTO_TEST_CASE(TestReArrange) {
  MatrixX<int> m(6, 3);
  m.row(0) << 1, 2, 3;
  m.row(1) << 4, 5, 6;
  m.row(2) << 7, 8, 9;
  m.row(3) << 10, 11, 12;
  m.row(4) << 13, 14, 15;
  m.row(5) << 16, 17, 18;

  VectorX<int> indices(3);
  indices << 1, 2, 0;

  const MatrixX<int> m2 = solvers::math::ReArrange(m, indices);

  MatrixX<int> comp(3, 3);
  comp.row(0) << 4, 5, 6;
  comp.row(1) << 7, 8, 9;
  comp.row(2) << 1, 2, 3;

  BOOST_REQUIRE(m2.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestReArrange2) {
  MatrixX<int> m(6, 3);
  m.row(0) << 1, 2, 3;
  m.row(1) << 4, 5, 6;
  m.row(2) << 7, 8, 9;
  m.row(3) << 10, 11, 12;
  m.row(4) << 13, 14, 15;
  m.row(5) << 16, 17, 18;

  VectorX<int> indices(4);
  indices << 1, 2, 0, 3;

  const MatrixX<int> m2 = solvers::math::ReArrange(m, indices);

  MatrixX<int> comp(4, 3);
  comp.row(0) << 4, 5, 6;
  comp.row(1) << 7, 8, 9;
  comp.row(2) << 1, 2, 3;
  comp.row(3) << 10, 11, 12;

  BOOST_REQUIRE(m2.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestIndexMatrixByMatrix) {
  MatrixX<int> m(6, 3);
  m.row(0) << 1, 2, 3;
  m.row(1) << 4, 5, 6;
  m.row(2) << 7, 8, 9;
  m.row(3) << 10, 11, 12;
  m.row(4) << 13, 14, 15;
  m.row(5) << 16, 17, 18;

  MatrixX<int> indices(3, 3);
  indices.row(0) << 0, 1, 2;
  indices.row(1) << 1, 2, 0;
  indices.row(2) << 4, 5, 2;

  const MatrixX<int> m2 = solvers::math::IndexMatrixByMatrix(m, indices);

  MatrixX<int> comp(9, 3);
  comp.row(0) << 1, 2, 3;
  comp.row(1) << 4, 5, 6;
  comp.row(2) << 7, 8, 9;

  comp.row(3) << 4, 5, 6;
  comp.row(4) << 7, 8, 9;
  comp.row(5) << 1, 2, 3;

  comp.row(6) << 13, 14, 15;
  comp.row(7) << 16, 17, 18;
  comp.row(8) << 7, 8, 9;

  BOOST_REQUIRE(m2.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestIndexMatrixByMatrix2) {
  MatrixXr m(6, 3);
  m.row(0) << 1, 2, 3;
  m.row(1) << 4, 5, 6;
  m.row(2) << 7, 8, 9;
  m.row(3) << 10, 11, 12;
  m.row(4) << 13, 14, 15;
  m.row(5) << 16, 17, 18;

  MatrixX<int> indices(3, 3);
  indices.row(0) << 0, 1, 2;
  indices.row(1) << 1, 2, 0;
  indices.row(2) << 4, 5, 2;

  const MatrixXr m2 = solvers::math::IndexMatrixByMatrix(m, indices);

  MatrixXr comp(9, 3);
  comp.row(0) << 1, 2, 3;
  comp.row(1) << 4, 5, 6;
  comp.row(2) << 7, 8, 9;

  comp.row(3) << 4, 5, 6;
  comp.row(4) << 7, 8, 9;
  comp.row(5) << 1, 2, 3;

  comp.row(6) << 13, 14, 15;
  comp.row(7) << 16, 17, 18;
  comp.row(8) << 7, 8, 9;

  BOOST_REQUIRE(m2.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestIndexMatrixByMatrixWithCol) {
  MatrixX<int> m(6, 3);
  m.row(0) << 1, 2, 3;
  m.row(1) << 4, 5, 6;
  m.row(2) << 7, 8, 9;
  m.row(3) << 10, 11, 12;
  m.row(4) << 13, 14, 15;
  m.row(5) << 16, 17, 18;

  MatrixX<int> indices(3, 3);
  indices.row(0) << 0, 1, 2;
  indices.row(1) << 1, 2, 0;
  indices.row(2) << 4, 5, 2;

  const MatrixX<int> m2 = solvers::math::IndexMatrixByMatrix(m, indices, 0);

  MatrixX<int> comp(3, 3);
  comp.row(0) << 1, 4, 7;
  comp.row(1) << 4, 7, 1;
  comp.row(2) << 13, 16, 7;

  BOOST_REQUIRE(m2.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestIndexMatrixByMatrixWithCol2) {
  MatrixXr m(6, 3);
  m.row(0) << 1, 2, 3;
  m.row(1) << 4, 5, 6;
  m.row(2) << 7, 8, 9;
  m.row(3) << 10, 11, 12;
  m.row(4) << 13, 14, 15;
  m.row(5) << 16, 17, 18;

  MatrixX<int> indices(3, 3);
  indices.row(0) << 0, 1, 2;
  indices.row(1) << 1, 2, 0;
  indices.row(2) << 4, 5, 2;

  const MatrixXr m2 = solvers::math::IndexMatrixByMatrix(m, indices, 0);

  MatrixXr comp(3, 3);
  comp.row(0) << 1, 4, 7;
  comp.row(1) << 4, 7, 1;
  comp.row(2) << 13, 16, 7;

  BOOST_REQUIRE(m2.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestFromStack) {
  MatrixXr m(2, 2);
  m.setConstant(1);

  std::vector<MatrixXr> stack{m, m, m, m};

  Tensor3r value = Tensor3r::FromStack(stack);

  BOOST_REQUIRE(value.Layer(0).isApprox(m));
  BOOST_REQUIRE(value.Layer(1).isApprox(m));
  BOOST_REQUIRE(value.Layer(2).isApprox(m));
  BOOST_REQUIRE(value.Layer(3).isApprox(m));
}

BOOST_AUTO_TEST_CASE(TestReplicate) {
  MatrixXr m(2, 2);
  m.setConstant(1);

  Tensor3r value = Tensor3r::Replicate(m, 4);

  BOOST_REQUIRE(value.Layer(0).isApprox(m));
  BOOST_REQUIRE(value.Layer(1).isApprox(m));
  BOOST_REQUIRE(value.Layer(2).isApprox(m));
  BOOST_REQUIRE(value.Layer(3).isApprox(m));
}

BOOST_AUTO_TEST_CASE(TestWhereIdx) {
  VectorX<int> comp(360);
  comp << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60,
      61, 70, 71, 80, 81, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
      103, 104, 105, 106, 107, 108, 109, 110, 111, 120, 121, 130, 131, 140, 141,
      150, 151, 160, 161, 170, 171, 180, 181, 190, 191, 192, 193, 194, 195, 196,
      197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211,
      220, 221, 230, 231, 240, 241, 250, 251, 260, 261, 270, 271, 280, 281, 290,
      291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305,
      306, 307, 308, 309, 310, 311, 320, 321, 330, 331, 340, 341, 350, 351, 360,
      361, 370, 371, 380, 381, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399,
      400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 420, 421, 430,
      431, 440, 441, 450, 451, 460, 461, 470, 471, 480, 481, 490, 491, 492, 493,
      494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508,
      509, 510, 511, 520, 521, 530, 531, 540, 541, 550, 551, 560, 561, 570, 571,
      580, 581, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602,
      603, 604, 605, 606, 607, 608, 609, 610, 611, 620, 621, 630, 631, 640, 641,
      650, 651, 660, 661, 670, 671, 680, 681, 690, 691, 692, 693, 694, 695, 696,
      697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711,
      720, 721, 730, 731, 740, 741, 750, 751, 760, 761, 770, 771, 780, 781, 790,
      791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805,
      806, 807, 808, 809, 810, 811, 820, 821, 830, 831, 840, 841, 850, 851, 860,
      861, 870, 871, 880, 881, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899,
      900, 901, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 920, 921, 930,
      931, 940, 941, 950, 951, 960, 961, 970, 971, 980, 981, 990, 991, 992, 993,
      994, 995, 996, 997, 998, 999, 1000;
  comp -= VectorX<int>::Ones(360);

  MatrixXr surface(10, 10);
  surface.row(0) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
  surface.row(1) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  surface.row(2) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  surface.row(3) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  surface.row(4) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  surface.row(5) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  surface.row(6) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  surface.row(7) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  surface.row(8) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
  surface.row(9) << 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;

  Tensor3r surface_mesh = Tensor3r::Replicate(surface, 10);
  VectorX<int> indices = surface_mesh.WhereIdx(1);

  BOOST_REQUIRE(indices.isApprox(comp));
}

BOOST_AUTO_TEST_CASE(TestNNZ) {
  VectorX<int> d = VectorX<int>::Zero(10);
  d(1) = 1;
  d(2) = 1;
  d(3) = 1;
  d(4) = 1;
  d(5) = 1;
  d(6) = 1;
  std::cout << d.transpose() << std::endl;

  VectorX<int> I;
  unsigned int c;
  solvers::math::NNZ(d, I, c, true);

  VectorX<int> IC(6);
  IC << 1, 2, 3, 4, 5, 6;
  BOOST_REQUIRE_EQUAL(c, 6);
  BOOST_REQUIRE(IC.isApprox(I));
}

BOOST_AUTO_TEST_CASE(TestComputeTriangleAngles) {
  const Vector3r a(1, 0, 0);
  const Vector3r b(4, 6, 0);
  const Vector3r c(-3, 5, 0);

  Real A_hat;
  Real B_hat;
  Real C_hat;

  solvers::math::ComputeTriangleAngles(a, b, c, A_hat, B_hat, C_hat);

  BOOST_REQUIRE(IsApprox(A_hat, 1.13, 0.01));
  BOOST_REQUIRE(IsApprox(B_hat, 0.96, 0.01));
  BOOST_REQUIRE(IsApprox(C_hat, 1.03, 0.01));
}

BOOST_AUTO_TEST_CASE(TestDegrees) {
    const Real denom = solvers::math::kPi / 4;
    BOOST_REQUIRE_EQUAL(solvers::math::Degrees(denom), 45);
}

BOOST_AUTO_TEST_CASE(TestRadians) {
  const Real denom = 90;
  BOOST_REQUIRE(solvers::math::IsApprox(solvers::math::Radians(denom), 1.570796, 0.001));
}
