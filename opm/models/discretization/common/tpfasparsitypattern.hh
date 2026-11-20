// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright Equinor ASA 2026

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

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 *
 * \brief Helpers to set up the matrix structure of a cell-centered (TPFA)
 *        discretization: the cell-to-cell sparsity pattern, the face neighbors of
 *        every cell, and the addresses of the matrix blocks that the assembly writes to.
 */
#ifndef OPM_TPFA_SPARSITY_PATTERN_HH
#define OPM_TPFA_SPARSITY_PATTERN_HH

#include <opm/common/utility/SparseTable.hpp>

#include <cassert>
#include <cstddef>
#include <set>
#include <vector>

namespace Opm {

using CellSparsityPattern = std::vector<std::set<unsigned>>;

/*!
 * \brief Sparsity pattern and face neighbors of a cell-centered discretization.
 */
struct CellConnectivity
{
    //! Row I holds I itself and all cells that share an interior face with I, sorted. This is
    //! the input of IstlSparseMatrixAdapter::reserve().
    CellSparsityPattern pattern;

    //! Row I holds the neighbors of I in the order of its interior faces in the stencil, i.e.
    //! neighbors[I][scvfIdx] is the cell on the other side of interior face scvfIdx. This is
    //! the same order as the neighborInfo and floresInfo of TpfaLinearizer.
    SparseTable<unsigned> neighbors;
};

/*!
 * \brief Build the sparsity pattern and the face neighbors of a cell-centered discretization.
 *
 * The grid is traversed in the same order as in TpfaLinearizer::createMatrix_(), and the
 * rows of the neighbor table are appended in that order. Like TpfaLinearizer, this assumes
 * that the cells are visited in the order of their indices.
 *
 * \param cellFn Called as cellFn(myIdx, stencil) once for every primary degree of freedom,
 *               after the stencil has been updated for its element. Callers can use it to
 *               collect per-face data (face area, half transmissibilities, ...) in the same
 *               traversal. The neighbor of interior face scvfIdx is
 *               stencil.globalSpaceIndex(scvfIdx + 1).
 */
template <class Stencil, class GridView, class DofMapper, class CellFn>
CellConnectivity buildCellConnectivity(const GridView& gridView,
                                       const DofMapper& dofMapper,
                                       const std::size_t numCells,
                                       CellFn&& cellFn)
{
    CellConnectivity connectivity;
    connectivity.pattern.resize(numCells);
    const int numRows = static_cast<int>(numCells); // SparseTable uses int sizes
    connectivity.neighbors.reserve(numRows, 6 * numRows);
    std::vector<unsigned> cellNeighbors;
    Stencil stencil(gridView, dofMapper);
    for (const auto& elem : elements(gridView)) {
        stencil.update(elem);
        for (unsigned primaryDofIdx = 0; primaryDofIdx < stencil.numPrimaryDof(); ++primaryDofIdx) {
            const unsigned myIdx = stencil.globalSpaceIndex(primaryDofIdx);
            assert(static_cast<int>(myIdx) == connectivity.neighbors.size());
            connectivity.pattern[myIdx].insert(myIdx);
            cellNeighbors.clear();
            for (unsigned dofIdx = 1; dofIdx < stencil.numDof(); ++dofIdx) {
                const unsigned neighborIdx = stencil.globalSpaceIndex(dofIdx);
                connectivity.pattern[myIdx].insert(neighborIdx);
                cellNeighbors.push_back(neighborIdx);
            }
            connectivity.neighbors.appendRow(cellNeighbors.begin(), cellNeighbors.end());
            cellFn(myIdx, stencil);
        }
    }
    return connectivity;
}

/*!
 * \brief Build the sparsity pattern and the face neighbors of a cell-centered discretization.
 *
 * Overload without a per-cell callback.
 */
template <class Stencil, class GridView, class DofMapper>
CellConnectivity buildCellConnectivity(const GridView& gridView,
                                       const DofMapper& dofMapper,
                                       const std::size_t numCells)
{
    return buildCellConnectivity<Stencil>(gridView, dofMapper, numCells,
                                          [](unsigned, const Stencil&) {});
}

/*!
 * \brief Addresses of the matrix blocks that the TPFA assembly of a cell writes to.
 *
 * The convention is the one of TpfaLinearizer: when cell I is assembled, the derivatives with
 * respect to the unknowns of I go into column I only, i.e. into diag[I] = &A[I][I] for the
 * equation of I, and into face[I][scvfIdx] = &A[J][I] for the equation of the neighbor J
 * across interior face scvfIdx. Different cells therefore never write to the same block,
 * which makes the cell loop thread safe.
 */
template <class MatrixBlock>
struct MatrixBlockAddresses
{
    std::vector<MatrixBlock*> diag;
    SparseTable<MatrixBlock*> face;
};

/*!
 * \brief Collect the block addresses of a matrix whose structure was set up from a
 *        CellConnectivity.
 *
 * \param matrix    An IstlSparseMatrixAdapter (or anything with blockAddress(row, col)),
 *                  after reserve() has been called.
 * \param neighbors The neighbors of CellConnectivity, in interior face order.
 */
template <class MatrixAdapter>
MatrixBlockAddresses<typename MatrixAdapter::MatrixBlock>
collectMatrixBlockAddresses(const MatrixAdapter& matrix,
                            const SparseTable<unsigned>& neighbors)
{
    using MatrixBlock = typename MatrixAdapter::MatrixBlock;
    const int numCells = neighbors.size();
    MatrixBlockAddresses<MatrixBlock> addresses;
    addresses.diag.resize(numCells);
    addresses.face.reserve(numCells, neighbors.dataSize());
    std::vector<MatrixBlock*> cellFaces;
    for (int globI = 0; globI < numCells; ++globI) {
        addresses.diag[globI] = matrix.blockAddress(globI, globI);
        cellFaces.clear();
        for (const unsigned globJ : neighbors[globI]) {
            cellFaces.push_back(matrix.blockAddress(globJ, globI));
        }
        addresses.face.appendRow(cellFaces.begin(), cellFaces.end());
    }
    return addresses;
}

} // namespace Opm

#endif // OPM_TPFA_SPARSITY_PATTERN_HH
