/*
  Copyright 2025 NORCE AS

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <config.h>

#define BOOST_TEST_MODULE TpsaMatrixTest

#include <boost/test/unit_test.hpp>

#include <opm/simulators/linalg/tpsa/TpsaMatrix.hpp>
#include <opm/simulators/linalg/tpsa/TpsaVector.hpp>

#include <dune/common/indices.hh>

#include <cstddef>
#include <set>
#include <vector>

namespace
{

using Scalar = double;
using Matrix = Opm::Linear::TpsaMatrix<Scalar>;
using Vector = Opm::Linear::TpsaVector<Scalar>;
using MatrixBlock = Matrix::MatrixBlock;
using EqVector = Vector::EqVector;

constexpr int numEq = Opm::Linear::numTpsaEq;
constexpr std::size_t numCells = 5;

//! \brief Tri-diagonal stencil, as produced by TpsaLinearizer for a 1-D grid.
std::vector<std::set<unsigned> >
makePattern()
{
    std::vector<std::set<unsigned> > pattern(numCells);
    for (unsigned i = 0; i < numCells; ++i) {
        pattern[i].insert(i);
        if (i > 0) {
            pattern[i].insert(i - 1);
        }
        if (i + 1 < numCells) {
            pattern[i].insert(i + 1);
        }
    }

    return pattern;
}

//! \brief A distinct value for every scalar of every block.
Scalar
entry(std::size_t row, std::size_t col, int eqIdx, int pvIdx)
{
    return 1000.0 * static_cast<Scalar>(row) + 100.0 * static_cast<Scalar>(col)
        + 10.0 * eqIdx + pvIdx + 1.0;
}

MatrixBlock
makeBlock(std::size_t row, std::size_t col)
{
    MatrixBlock b(0.0);
    for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
        for (int pvIdx = 0; pvIdx < numEq; ++pvIdx) {
            b[eqIdx][pvIdx] = entry(row, col, eqIdx, pvIdx);
        }
    }

    return b;
}

//! \brief True for the six displacement-displacement couplings that are dropped.
bool
isDropped(int eqIdx, int pvIdx)
{
    return eqIdx < 3 && pvIdx < 3 && eqIdx != pvIdx;
}

//! \brief Assemble the test matrix the way the linearizer would.
void
assemble(Matrix& m, const std::vector<std::set<unsigned> >& pattern)
{
    for (std::size_t row = 0; row < numCells; ++row) {
        for (const auto col : pattern[row]) {
            auto address = m.blockAddress(row, col);
            const auto b = makeBlock(row, col);
            // Same syntax the linearizer uses on a MatrixBlock*.
            *address += b;
        }
    }
}

} // anonymous namespace

BOOST_AUTO_TEST_CASE(ScatterMatchesTheFieldSplit)
{
    const auto pattern = makePattern();
    Matrix m(numCells, numCells);
    m.reserve(pattern);
    assemble(m, pattern);

    const auto& view = m.istlMatrix();
    using namespace Dune::Indices;

    for (std::size_t row = 0; row < numCells; ++row) {
        for (const auto col : pattern[row]) {
            // Diagonal displacement blocks: 1x1, one per component.
            BOOST_CHECK_EQUAL(view[_0][_0][row][col][0][0], entry(row, col, 0, 0));
            BOOST_CHECK_EQUAL(view[_1][_1][row][col][0][0], entry(row, col, 1, 1));
            BOOST_CHECK_EQUAL(view[_2][_2][row][col][0][0], entry(row, col, 2, 2));

            for (int j = 0; j < 3; ++j) {
                // Displacement-rotation: 1x3
                BOOST_CHECK_EQUAL(view[_0][_3][row][col][0][j], entry(row, col, 0, 3 + j));
                BOOST_CHECK_EQUAL(view[_1][_3][row][col][0][j], entry(row, col, 1, 3 + j));
                BOOST_CHECK_EQUAL(view[_2][_3][row][col][0][j], entry(row, col, 2, 3 + j));

                // Rotation-displacement: 3x1
                BOOST_CHECK_EQUAL(view[_3][_0][row][col][j][0], entry(row, col, 3 + j, 0));
                BOOST_CHECK_EQUAL(view[_3][_1][row][col][j][0], entry(row, col, 3 + j, 1));
                BOOST_CHECK_EQUAL(view[_3][_2][row][col][j][0], entry(row, col, 3 + j, 2));

                // Rotation-solid pressure and solid pressure-rotation
                BOOST_CHECK_EQUAL(view[_3][_4][row][col][j][0], entry(row, col, 3 + j, 6));
                BOOST_CHECK_EQUAL(view[_4][_3][row][col][0][j], entry(row, col, 6, 3 + j));

                for (int i = 0; i < 3; ++i) {
                    BOOST_CHECK_EQUAL(view[_3][_3][row][col][i][j],
                                      entry(row, col, 3 + i, 3 + j));
                }
            }

            // Displacement-solid pressure and solid pressure-displacement
            BOOST_CHECK_EQUAL(view[_0][_4][row][col][0][0], entry(row, col, 0, 6));
            BOOST_CHECK_EQUAL(view[_1][_4][row][col][0][0], entry(row, col, 1, 6));
            BOOST_CHECK_EQUAL(view[_2][_4][row][col][0][0], entry(row, col, 2, 6));
            BOOST_CHECK_EQUAL(view[_4][_0][row][col][0][0], entry(row, col, 6, 0));
            BOOST_CHECK_EQUAL(view[_4][_1][row][col][0][0], entry(row, col, 6, 1));
            BOOST_CHECK_EQUAL(view[_4][_2][row][col][0][0], entry(row, col, 6, 2));

            BOOST_CHECK_EQUAL(view[_4][_4][row][col][0][0], entry(row, col, 6, 6));
        }
    }
}

BOOST_AUTO_TEST_CASE(GatherDropsDisplacementCrossCouplings)
{
    const auto pattern = makePattern();
    Matrix m(numCells, numCells);
    m.reserve(pattern);
    assemble(m, pattern);

    // Reading a block back gives the assembled values, except for the six
    // displacement-displacement off-diagonals, which are not stored at all.
    for (std::size_t row = 0; row < numCells; ++row) {
        for (const auto col : pattern[row]) {
            MatrixBlock b;
            m.block(row, col, b);
            for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
                for (int pvIdx = 0; pvIdx < numEq; ++pvIdx) {
                    const Scalar expected = isDropped(eqIdx, pvIdx)
                        ? 0.0
                        : entry(row, col, eqIdx, pvIdx);
                    BOOST_CHECK_EQUAL(b[eqIdx][pvIdx], expected);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(MatrixVectorProductMatchesDenseReference)
{
    const auto pattern = makePattern();
    Matrix m(numCells, numCells);
    m.reserve(pattern);
    assemble(m, pattern);

    Vector x(numCells);
    for (std::size_t i = 0; i < numCells; ++i) {
        for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
            x[i][eqIdx] = 0.5 * static_cast<Scalar>(i) + 0.25 * eqIdx + 1.0;
        }
    }

    Vector y(numCells);
    y = 0.0;
    m.istlMatrix().mv(x.istlVector(), y.istlVector());

    // Reference: dense multiply with the dropped entries set to zero.
    for (std::size_t row = 0; row < numCells; ++row) {
        EqVector expected(0.0);
        for (const auto col : pattern[row]) {
            for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
                for (int pvIdx = 0; pvIdx < numEq; ++pvIdx) {
                    if (isDropped(eqIdx, pvIdx)) {
                        continue;
                    }
                    expected[eqIdx] += entry(row, col, eqIdx, pvIdx)
                        * (0.5 * static_cast<Scalar>(col) + 0.25 * pvIdx + 1.0);
                }
            }
        }

        const EqVector actual = static_cast<const Vector&>(y)[row];
        for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
            BOOST_CHECK_CLOSE(actual[eqIdx], expected[eqIdx], 1e-10);
        }
    }
}

BOOST_AUTO_TEST_CASE(ClearAndClearRow)
{
    const auto pattern = makePattern();
    Matrix m(numCells, numCells);
    m.reserve(pattern);
    assemble(m, pattern);

    constexpr std::size_t row = 2;
    m.clearRow(row, 1.0);

    MatrixBlock b;
    for (const auto col : pattern[row]) {
        m.block(row, col, b);
        for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
            for (int pvIdx = 0; pvIdx < numEq; ++pvIdx) {
                const bool onDiagonal = (col == row) && (eqIdx == pvIdx);
                BOOST_CHECK_EQUAL(b[eqIdx][pvIdx], onDiagonal ? 1.0 : 0.0);
            }
        }
    }

    // Other rows are untouched.
    m.block(1, 1, b);
    BOOST_CHECK_EQUAL(b[0][0], entry(1, 1, 0, 0));

    m.clear();
    m.block(1, 1, b);
    for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
        for (int pvIdx = 0; pvIdx < numEq; ++pvIdx) {
            BOOST_CHECK_EQUAL(b[eqIdx][pvIdx], 0.0);
        }
    }
}

BOOST_AUTO_TEST_CASE(HypreCompatibleDiagonalBlocks)
{
    // Hypre's BoomerAMG is only registered for 1x1-block matrices whose values
    // live in one contiguous array reachable through &A[0][0][0][0].
    using DD00 = std::decay_t<decltype(std::declval<Matrix&>().dd00())>;
    using SPSP = std::decay_t<decltype(std::declval<Matrix&>().spsp())>;
    static_assert(DD00::block_type::rows == 1 && DD00::block_type::cols == 1);
    static_assert(SPSP::block_type::rows == 1 && SPSP::block_type::cols == 1);

    const auto pattern = makePattern();
    Matrix m(numCells, numCells);
    m.reserve(pattern);
    assemble(m, pattern);

    const Scalar* values = &(m.dd00()[0][0][0][0]);
    std::size_t k = 0;
    for (std::size_t row = 0; row < numCells; ++row) {
        for (const auto col : pattern[row]) {
            BOOST_CHECK_EQUAL(values[k], entry(row, col, 0, 0));
            ++k;
        }
    }
    BOOST_CHECK_EQUAL(k, m.nonzeroes());
}

BOOST_AUTO_TEST_CASE(VectorRoundTrip)
{
    Vector v(numCells);
    v = 0.0;

    for (std::size_t i = 0; i < numCells; ++i) {
        EqVector contribution;
        for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
            contribution[eqIdx] = 10.0 * static_cast<Scalar>(i) + eqIdx;
        }
        v[i] += contribution;
        v[i] += contribution;
    }

    const Vector& constView = v;
    for (std::size_t i = 0; i < numCells; ++i) {
        const EqVector block = constView[i];
        BOOST_CHECK_EQUAL(block.size(), static_cast<std::size_t>(numEq));
        for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
            BOOST_CHECK_CLOSE(block[eqIdx], 2 * (10.0 * static_cast<Scalar>(i) + eqIdx), 1e-12);
        }
    }

    // Sub-vectors carry the fields in the expected order.
    using namespace Dune::Indices;
    BOOST_CHECK_CLOSE(v.istlVector()[_0][3][0], 2 * 30.0, 1e-12);
    BOOST_CHECK_CLOSE(v.istlVector()[_3][3][1], 2 * (30.0 + 4), 1e-12);
    BOOST_CHECK_CLOSE(v.istlVector()[_4][3][0], 2 * (30.0 + 6), 1e-12);

    Scalar expectedOneNorm = 0.0;
    for (std::size_t i = 0; i < numCells; ++i) {
        for (int eqIdx = 0; eqIdx < numEq; ++eqIdx) {
            expectedOneNorm += 2 * (10.0 * static_cast<Scalar>(i) + eqIdx);
        }
    }
    BOOST_CHECK_CLOSE(v.one_norm(), expectedOneNorm, 1e-12);

    v[2] = 0.0;
    BOOST_CHECK_EQUAL(constView[2][4], 0.0);
}
