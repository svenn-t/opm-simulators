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
/**
 * \file
 *
 * \copydoc Opm::TransportLinearSystem
 */
#ifndef OPM_TRANSPORT_LINEAR_SYSTEM_HPP
#define OPM_TRANSPORT_LINEAR_SYSTEM_HPP

#include <dune/grid/common/gridenums.hh>
#include <dune/grid/common/rangegenerators.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>

#include <opm/common/utility/SparseTable.hpp>

#include <opm/models/discretization/common/tpfalinearizerstructs.hh>

#include <opm/simulators/linalg/istlsparsematrixadapter.hh>
#include <opm/simulators/linalg/matrixblock.hh>

#include <cstddef>
#include <memory>
#include <set>
#include <vector>

namespace Opm {

/*!
 * \brief Cell-centred linear system for sequentially solved, tracer-like
 *        transport equations (passive tracers, sequential temperature, ...).
 *
 * The class owns the Jacobian matrix, the table of interior faces of each cell
 * and pointers to the matrix blocks, so the models can assemble their equations
 * by looping over cells and faces without an ElementContext, in the same way
 * as the TpfaLinearizer does for the flow equations.
 *
 * Each model attaches its own static data to the faces (\c FaceInfo). The
 * boundary faces of each cell are recorded as well, for models with boundary
 * conditions.
 *
 * \tparam Scalar    Floating point type.
 * \tparam numEq     Number of equations per cell (block size of the matrix).
 * \tparam FaceInfo  Static per-face data stored in the neighbor table.
 */
template <class Scalar, int numEq, class FaceInfo>
class TransportLinearSystem
{
public:
    using MatrixBlock = Opm::MatrixBlock<Scalar, numEq, numEq>;
    using Matrix = Dune::BCRSMatrix<MatrixBlock>;
    using Vector = Dune::BlockVector<Dune::FieldVector<Scalar, numEq>>;
    using NeighborInfo = NeighborInfoStruct<FaceInfo, MatrixBlock>;

    //! \brief A face of a cell on the boundary of the domain.
    struct BoundaryFace
    {
        unsigned bfIdx{}; //!< Index of the boundary face in the stencil of the cell
        int dirId{};      //!< Direction of the face, negative for NNCs
        Scalar area{};
        Scalar zCoord{};  //!< Depth of the face centre
    };

    /*!
     * \brief Set up the neighbor table and the matrix.
     *
     * \param faceInfo Callable with signature
     *        FaceInfo(unsigned globI, unsigned globJ, const InteriorFace& scvf)
     *        returning the static data of the face between cell globI and globJ.
     */
    template <class Stencil, class GridView, class DofMapper, class FaceInfoFunc>
    void init(const GridView& gridView,
              const DofMapper& dofMapper,
              const unsigned numCells,
              FaceInfoFunc&& faceInfo)
    {
        std::vector<std::set<unsigned>> sparsityPattern(numCells);
        isInterior_.assign(numCells, false);
        neighborInfo_.clear();
        neighborInfo_.reserve(numCells, 6 * numCells);
        boundaryFaces_.clear();

        Stencil stencil(gridView, dofMapper);
        std::vector<NeighborInfo> locNbInfo;
        std::vector<BoundaryFace> locBoundaryFaces;
        for (const auto& elem : elements(gridView)) {
            stencil.update(elem);
            for (unsigned primaryDofIdx = 0; primaryDofIdx < stencil.numPrimaryDof(); ++primaryDofIdx) {
                const unsigned myIdx = stencil.globalSpaceIndex(primaryDofIdx);
                isInterior_[myIdx] = elem.partitionType() == Dune::InteriorEntity;
                locNbInfo.resize(stencil.numDof() - 1); // the primary dof is not a neighbor
                for (unsigned dofIdx = 0; dofIdx < stencil.numDof(); ++dofIdx) {
                    const unsigned neighborIdx = stencil.globalSpaceIndex(dofIdx);
                    sparsityPattern[myIdx].insert(neighborIdx);
                    if (dofIdx > 0) {
                        const auto& scvf = stencil.interiorFace(dofIdx - 1);
                        locNbInfo[dofIdx - 1] =
                            NeighborInfo{neighborIdx, faceInfo(myIdx, neighborIdx, scvf), nullptr};
                    }
                }
                neighborInfo_.appendRow(locNbInfo.begin(), locNbInfo.end());

                locBoundaryFaces.clear();
                for (unsigned bfIdx = 0; bfIdx < stencil.numBoundaryFaces(); ++bfIdx) {
                    const auto& bf = stencil.boundaryFace(bfIdx);
                    locBoundaryFaces.push_back({bfIdx,
                                                bf.dirId(),
                                                static_cast<Scalar>(bf.area()),
                                                static_cast<Scalar>(bf.integrationPos()[GridView::dimensionworld - 1])});
                }
                boundaryFaces_.appendRow(locBoundaryFaces.begin(), locBoundaryFaces.end());
            }
        }

        matrix_ = std::make_unique<SparseAdapter>(numCells, numCells);
        matrix_->reserve(sparsityPattern);
        diagMatAddress_.resize(numCells);
        for (unsigned globI = 0; globI < numCells; ++globI) {
            diagMatAddress_[globI] = matrix_->blockAddress(globI, globI);
            for (auto& nbInfo : neighborInfo_[globI]) {
                // Derivative of the residual of the neighbor with respect to globI
                nbInfo.matBlockAddress = matrix_->blockAddress(nbInfo.neighbor, globI);
            }
        }
    }

    //! \brief Whether init() has been called.
    bool initialized() const
    { return matrix_ != nullptr; }

    std::size_t numCells() const
    { return diagMatAddress_.size(); }

    //! \brief Interior faces of a cell, see NeighborInfoStruct.
    auto neighbors(const unsigned globI) const
    { return neighborInfo_[globI]; }

    //! \brief Boundary faces of a cell, see BoundaryFace.
    auto boundaryFaces(const unsigned globI) const
    { return boundaryFaces_[globI]; }

    //! \brief Diagonal block of the matrix for a cell.
    MatrixBlock& diagonal(const unsigned globI)
    { return *diagMatAddress_[globI]; }

    //! \brief Whether the cell is in the interior partition of this process.
    bool isInterior(const unsigned globI) const
    { return isInterior_[globI]; }

    Matrix& matrix()
    { return matrix_->istlMatrix(); }

    const Matrix& matrix() const
    { return matrix_->istlMatrix(); }

    //! \brief Set all matrix entries to zero.
    void clearMatrix()
    { matrix_->clear(); }

private:
    using SparseAdapter = Linear::IstlSparseMatrixAdapter<MatrixBlock>;

    SparseTable<NeighborInfo> neighborInfo_{};
    SparseTable<BoundaryFace> boundaryFaces_{};
    std::vector<MatrixBlock*> diagMatAddress_{};
    std::vector<bool> isInterior_{};
    std::unique_ptr<SparseAdapter> matrix_{};
};

} // namespace Opm

#endif // OPM_TRANSPORT_LINEAR_SYSTEM_HPP
