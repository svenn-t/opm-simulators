// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
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
 * \brief Helper to build the cell-to-cell sparsity pattern of a cell-centered
 *        (TPFA) discretization.
 */
#ifndef OPM_TPFA_SPARSITY_PATTERN_HH
#define OPM_TPFA_SPARSITY_PATTERN_HH

#include <cstddef>
#include <set>
#include <vector>

namespace Opm {

using CellSparsityPattern = std::vector<std::set<unsigned>>;

/*!
 * \brief Build the cell-to-cell sparsity pattern of a cell-centered discretization.
 *
 * Row I contains I itself and all cells that share an interior face with I. The grid is
 * traversed in the same order as in TpfaLinearizer::createMatrix_().
 *
 * \param cellFn Called as cellFn(myIdx, stencil) once for every primary degree of freedom,
 *               after the stencil has been updated for its element. Callers can use it to
 *               collect per-face data (face area, half transmissibilities, ...) in the same
 *               traversal. The neighbor of interior face scvfIdx is
 *               stencil.globalSpaceIndex(scvfIdx + 1).
 */
template <class Stencil, class GridView, class DofMapper, class CellFn>
CellSparsityPattern buildCellSparsityPattern(const GridView& gridView,
                                             const DofMapper& dofMapper,
                                             const std::size_t numCells,
                                             CellFn&& cellFn)
{
    CellSparsityPattern pattern(numCells);
    Stencil stencil(gridView, dofMapper);
    for (const auto& elem : elements(gridView)) {
        stencil.update(elem);
        for (unsigned primaryDofIdx = 0; primaryDofIdx < stencil.numPrimaryDof(); ++primaryDofIdx) {
            const unsigned myIdx = stencil.globalSpaceIndex(primaryDofIdx);
            for (unsigned dofIdx = 0; dofIdx < stencil.numDof(); ++dofIdx) {
                pattern[myIdx].insert(stencil.globalSpaceIndex(dofIdx));
            }
            cellFn(myIdx, stencil);
        }
    }
    return pattern;
}

/*!
 * \brief Build the cell-to-cell sparsity pattern of a cell-centered discretization.
 *
 * Overload without a per-cell callback.
 */
template <class Stencil, class GridView, class DofMapper>
CellSparsityPattern buildCellSparsityPattern(const GridView& gridView,
                                             const DofMapper& dofMapper,
                                             const std::size_t numCells)
{
    return buildCellSparsityPattern<Stencil>(gridView, dofMapper, numCells,
                                             [](unsigned, const Stencil&) {});
}

} // namespace Opm

#endif // OPM_TPFA_SPARSITY_PATTERN_HH
