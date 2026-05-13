// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
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

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
#ifndef MATRIX_RESIDUAL_SPLITTER_HYPRE_HPP
#define MATRIX_RESIDUAL_SPLITTER_HYPRE_HPP

#include "SystemTypes.hpp"

namespace Opm
{
template <typename Scalar, typename Matrix, typename Vector>
class MatrixResidualSplitterHypreTPSA
{
public:
    MatrixResidualSplitterHypreTPSA() = default;

    MatrixResidualSplitterHypreTPSA(const Matrix& M,
                                    const Vector& residual,
                                    const std::vector<std::set<unsigned> >& sparsityPattern)
        : parentMat_(&M)
          , parentRes_(&residual)
          , sparsityPattern_(sparsityPattern)
    {
    }

    void generateSubmatrices()
    {
        // Initialize and assign submatrices
        initializeSubMatrices_();
        assignSubMatrices(*DDmat_,
                          *DRmat_,
                          *DSmat_,
                          *RDmat_,
                          *RRmat_,
                          *RSmat_,
                          *SDmat_,
                          *SRmat_,
                          *SSmat_);

    }

    void assignSubMatrices(DispDispMatrixT<Scalar>& DDmat,
                           DispRotMatrixT<Scalar>& DRmat,
                           DispSPresMatrixT<Scalar>& DSmat,
                           RotDispMatrixT<Scalar>& RDmat,
                           RotRotMatrixT<Scalar>& RRmat,
                           RotSPresMatrixT<Scalar>& RSmat,
                           SPresDispMatrixT<Scalar>& SDmat,
                           SPresRotMatrixT<Scalar>& SRmat,
                           SPresSPresMatrixT<Scalar>& SSmat)
    {
        if (parentMat_ == nullptr) {
            return;
        }

        // Loop over parent matrix and assign entries to correct sub-matrix
        for (auto row = parentMat_->begin(); row != parentMat_->end(); ++row) {
            for (auto col = row->begin(); col != row->end(); ++col) {
                // Get parent block
                const auto& parentBlock = *col;

                // Assign to sub-matrices
                // Displacement-displacement matrices
                auto* DDMatBlock00 = DDmat.blockAddress(row.index() * 3, col.index() * 3);
                auto* DDMatBlock01 = DDmat.blockAddress(row.index() * 3, col.index() * 3 + 1);
                auto* DDMatBlock02 = DDmat.blockAddress(row.index() * 3, col.index() * 3 + 2);
                auto* DDMatBlock10 = DDmat.blockAddress(row.index() * 3 + 1, col.index() * 3);
                auto* DDMatBlock11 = DDmat.blockAddress(row.index() * 3 + 1, col.index() * 3 + 1);
                auto* DDMatBlock12 = DDmat.blockAddress(row.index() * 3 + 1, col.index() * 3 + 2);
                auto* DDMatBlock20 = DDmat.blockAddress(row.index() * 3 + 2, col.index() * 3);
                auto* DDMatBlock21 = DDmat.blockAddress(row.index() * 3 + 2, col.index() * 3 + 1);
                auto* DDMatBlock22 = DDmat.blockAddress(row.index() * 3 + 2, col.index() * 3 + 2);
                (*DDMatBlock00)[0][0] = parentBlock[0][0];
                (*DDMatBlock01)[0][0] = parentBlock[0][1];
                (*DDMatBlock02)[0][0] = parentBlock[0][2];
                (*DDMatBlock10)[0][0] = parentBlock[1][0];
                (*DDMatBlock11)[0][0] = parentBlock[1][1];
                (*DDMatBlock12)[0][0] = parentBlock[1][2];
                (*DDMatBlock20)[0][0] = parentBlock[2][0];
                (*DDMatBlock21)[0][0] = parentBlock[2][1];
                (*DDMatBlock22)[0][0] = parentBlock[2][2];

                // Displacement-rotation matrix
                auto* DRMatBlock0 = DRmat.blockAddress(row.index() * 3, col.index());
                auto* DRMatBlock1 = DRmat.blockAddress(row.index() * 3 + 1, col.index());
                auto* DRMatBlock2 = DRmat.blockAddress(row.index() * 3 + 2, col.index());
                (*DRMatBlock0)[0][0] = parentBlock[0][3];
                (*DRMatBlock0)[0][1] = parentBlock[0][4];
                (*DRMatBlock0)[0][2] = parentBlock[0][5];
                (*DRMatBlock1)[0][0] = parentBlock[1][3];
                (*DRMatBlock1)[0][1] = parentBlock[1][4];
                (*DRMatBlock1)[0][2] = parentBlock[1][5];
                (*DRMatBlock2)[0][0] = parentBlock[2][3];
                (*DRMatBlock2)[0][1] = parentBlock[2][4];
                (*DRMatBlock2)[0][2] = parentBlock[2][5];

                // Displacement-solid pressure matrix
                auto* DSMatBlock0 = DSmat.blockAddress(row.index() * 3, col.index());
                auto* DSMatBlock1 = DSmat.blockAddress(row.index() * 3 + 1, col.index());
                auto* DSMatBlock2 = DSmat.blockAddress(row.index() * 3 + 2, col.index());
                (*DSMatBlock0)[0][0] = parentBlock[0][6];
                (*DSMatBlock1)[0][0] = parentBlock[1][6];
                (*DSMatBlock2)[0][0] = parentBlock[2][6];

                // Rotation-displacement matrix
                auto* RDMatBlock0 = RDmat.blockAddress(row.index(), col.index() * 3);
                auto* RDMatBlock1 = RDmat.blockAddress(row.index(), col.index() * 3 + 1);
                auto* RDMatBlock2 = RDmat.blockAddress(row.index(), col.index() * 3 + 1);
                (*RDMatBlock0)[0][0] = parentBlock[3][0];
                (*RDMatBlock1)[0][0] = parentBlock[3][1];
                (*RDMatBlock2)[0][0] = parentBlock[3][2];
                (*RDMatBlock0)[1][0] = parentBlock[4][0];
                (*RDMatBlock1)[1][0] = parentBlock[4][1];
                (*RDMatBlock2)[1][0] = parentBlock[4][2];
                (*RDMatBlock0)[2][0] = parentBlock[5][0];
                (*RDMatBlock1)[2][0] = parentBlock[5][1];
                (*RDMatBlock2)[2][0] = parentBlock[5][2];

                // Rotation-rotation matrix
                auto* RRMatBlock = RRmat.blockAddress(row.index(), col.index());
                (*RRMatBlock)[0][0] = parentBlock[3][3];
                (*RRMatBlock)[0][1] = parentBlock[3][4];
                (*RRMatBlock)[0][2] = parentBlock[3][5];
                (*RRMatBlock)[1][0] = parentBlock[4][3];
                (*RRMatBlock)[1][1] = parentBlock[4][4];
                (*RRMatBlock)[1][2] = parentBlock[4][5];
                (*RRMatBlock)[2][0] = parentBlock[5][3];
                (*RRMatBlock)[2][1] = parentBlock[5][4];
                (*RRMatBlock)[2][2] = parentBlock[5][5];

                // Rotation-solid pressure matrix
                auto* RSMatBlock = RSmat.blockAddress(row.index(), col.index());
                (*RSMatBlock)[0][0] = parentBlock[3][6];
                (*RSMatBlock)[1][0] = parentBlock[4][6];
                (*RSMatBlock)[2][0] = parentBlock[5][6];

                // Solid pressure-displacement matrix
                auto* SDMatBlock0 = SDmat.blockAddress(row.index(), col.index() * 3);
                auto* SDMatBlock1 = SDmat.blockAddress(row.index(), col.index() * 3 + 1);
                auto* SDMatBlock2 = SDmat.blockAddress(row.index(), col.index() * 3 + 2);
                (*SDMatBlock0)[0][0] = parentBlock[6][0];
                (*SDMatBlock1)[0][0] = parentBlock[6][1];
                (*SDMatBlock2)[0][0] = parentBlock[6][2];

                // Solid pressure-rotation matrix
                auto* SRMatBlock = SRmat.blockAddress(row.index(), col.index());
                (*SRMatBlock)[0][0] = parentBlock[6][3];
                (*SRMatBlock)[0][1] = parentBlock[6][4];
                (*SRMatBlock)[0][2] = parentBlock[6][5];

                // Solid pressure-solid pressure matrix
                auto* SSMatBlock = SSmat.blockAddress(row.index(), col.index());
                (*SSMatBlock)[0][0] = parentBlock[6][6];
            }
        }
    }

    void generateSubResiduals()
    {
        // Initialize and assign subresiduals
        initializeSubResiduals_();
        assignSubResiduals(DRes_, RRes_, SRes_);
    }

    void assignSubResiduals(DispVectorT<Scalar>& DRes,
                            RotVectorT<Scalar>& RRes,
                            SPresVectorT<Scalar>& SRes)
    {
        if (parentRes_ == nullptr) {
            return;
        }

        // Assign sub-residuals
        for (std::size_t i = 0; i < parentRes_->size(); ++i) {
            DRes[3 * i][0] = (*parentRes_)[i][0];
            DRes[3 * i + 1][0] = (*parentRes_)[i][1];
            DRes[3 * i + 2][0] = (*parentRes_)[i][2];

            RRes[i][0] = (*parentRes_)[i][3];
            RRes[i][1] = (*parentRes_)[i][4];
            RRes[i][2] = (*parentRes_)[i][5];

            SRes[i][0] = (*parentRes_)[i][6];
        }
    }

    //
    // Transfer ownership
    //
    std::unique_ptr<DispDispMatrixT<Scalar> > takeDispDispMatrix() noexcept
    {
        return std::move(DDmat_);
    }

    std::unique_ptr<DispRotMatrixT<Scalar> > takeDispRotMatrix() noexcept
    {
        return std::move(DRmat_);
    }

    std::unique_ptr<DispSPresMatrixT<Scalar> > takeDispSPresMatrix() noexcept
    {
        return std::move(DSmat_);
    }

    std::unique_ptr<RotDispMatrixT<Scalar> > takeRotDispMatrix() noexcept
    {
        return std::move(RDmat_);
    }

    std::unique_ptr<RotRotMatrixT<Scalar> > takeRotRotMatrix() noexcept
    {
        return std::move(RRmat_);
    }

    std::unique_ptr<RotSPresMatrixT<Scalar> > takeRotSPresMatrix() noexcept
    {
        return std::move(RSmat_);
    }

    std::unique_ptr<SPresDispMatrixT<Scalar> > takeSPresDispMatrix() noexcept
    {
        return std::move(SDmat_);
    }

    std::unique_ptr<SPresRotMatrixT<Scalar> > takeSPresRotMatrix() noexcept
    {
        return std::move(SRmat_);
    }

    std::unique_ptr<SPresSPresMatrixT<Scalar> > takeSPresSPresMatrix() noexcept
    {
        return std::move(SSmat_);
    }

    DispVectorT<Scalar> takeDispVector() noexcept
    {
        return std::move(DRes_);
    }

    RotVectorT<Scalar> takeRotVector() noexcept
    {
        return std::move(RRes_);
    }

    SPresVectorT<Scalar> takeSPresVector() noexcept
    {
        return std::move(SRes_);
    }

    //
    //  Viewers
    //
    DispDispMatrixT<Scalar>& dispDispMatrix()
    {
        return *DDmat_;
    }

    const DispDispMatrixT<Scalar>& dispDispMatrix() const
    {
        return *DDmat_;
    }

    DispRotMatrixT<Scalar>& dispRotMatrix()
    {
        return *DRmat_;
    }

    const DispRotMatrixT<Scalar>& dispRotMatrix() const
    {
        return *DRmat_;
    }

    DispSPresMatrixT<Scalar>& dispSPresMatrix()
    {
        return *DSmat_;
    }

    const DispSPresMatrixT<Scalar>& dispSPresMatrix() const
    {
        return *DSmat_;
    }

    RotDispMatrixT<Scalar>& rotDispMatrix()
    {
        return *RDmat_;
    }

    const RotDispMatrixT<Scalar>& rotDispMatrix() const
    {
        return *RDmat_;
    }

    RotRotMatrixT<Scalar>& rotRotMatrix()
    {
        return *RRmat_;
    }

    const RotRotMatrixT<Scalar>& rotRotMatrix() const
    {
        return *RRmat_;
    }

    RotSPresMatrixT<Scalar>& rotSPresMatrix()
    {
        return *RSmat_;
    }

    const RotSPresMatrixT<Scalar>& rotSPresMatrix() const
    {
        return *RSmat_;
    }

    SPresDispMatrixT<Scalar>& sPresDispMatrix()
    {
        return *SDmat_;
    }

    const SPresDispMatrixT<Scalar>& sPresDispMatrix() const
    {
        return *SDmat_;
    }

    SPresRotMatrixT<Scalar>& sPresRotMatrix()
    {
        return *SRmat_;
    }

    const SPresRotMatrixT<Scalar>& sPresRotMatrix() const
    {
        return *SRmat_;
    }

    SPresSPresMatrixT<Scalar>& sPresSPresMatrix()
    {
        return *SSmat_;
    }

    const SPresSPresMatrixT<Scalar>& sPresSPresMatrix() const
    {
        return *SSmat_;
    }

    const DispVectorT<Scalar>& dispVector() const
    {
        return DRes_;
    }

    const RotVectorT<Scalar>& rotVector() const
    {
        return RRes_;
    }

    const SPresVectorT<Scalar>& sPresVector() const
    {
        return SRes_;
    }

private:
    void initializeSubMatrices_()
    {
        if (parentMat_ == nullptr || sparsityPattern_.empty()) {
            return;
        }

        // Instantiate sub-matrices
        DDmat_ = std::make_unique<DispDispMatrixT<Scalar> >(parentMat_->N() * 3,
                                                            parentMat_->M() * 3);
        DRmat_ = std::make_unique<DispRotMatrixT<Scalar> >(parentMat_->N() * 3,
                                                           parentMat_->M());
        DSmat_ = std::make_unique<DispSPresMatrixT<Scalar> >(parentMat_->N() * 3,
                                                             parentMat_->M());

        RDmat_ = std::make_unique<RotDispMatrixT<Scalar> >(parentMat_->N(),
                                                           parentMat_->M() * 3);
        RRmat_ = std::make_unique<RotRotMatrixT<Scalar> >(parentMat_->N(),
                                                          parentMat_->M());
        RSmat_ = std::make_unique<RotSPresMatrixT<Scalar> >(parentMat_->N(),
                                                            parentMat_->M());

        SDmat_ = std::make_unique<SPresDispMatrixT<Scalar> >(parentMat_->N(),
                                                             parentMat_->M() * 3);
        SRmat_ = std::make_unique<SPresRotMatrixT<Scalar> >(parentMat_->N(),
                                                            parentMat_->M());
        SSmat_ = std::make_unique<SPresSPresMatrixT<Scalar> >(parentMat_->N(),
                                                              parentMat_->M());

        // Set sparsity pattern
        RRmat_->reserve(sparsityPattern_);
        RSmat_->reserve(sparsityPattern_);

        SRmat_->reserve(sparsityPattern_);
        SSmat_->reserve(sparsityPattern_);

        reserveDDmatDRmatDSmat_();
        reserveRDmatSDmat();
    }

    void reserveDDmatDRmatDSmat_()
    {
        DDmat_->resetIstlMatrix();
        DRmat_->resetIstlMatrix();
        DSmat_->resetIstlMatrix();
        auto& DDmat = DDmat_->istlMatrix();
        auto& DRmat = DRmat_->istlMatrix();
        auto& DSmat = DSmat_->istlMatrix();

        assert(DDmat_->rows() == sparsityPattern_.size() * 3);
        assert(DRmat_->rows() == sparsityPattern_.size() * 3);
        assert(DSmat_->rows() == sparsityPattern_.size() * 3);

        for (std::size_t dofIdx = 0; dofIdx < DDmat_->rows(); ++dofIdx) {
            const auto spIdx = dofIdx / 3;
            DDmat.setrowsize(dofIdx, sparsityPattern_[spIdx].size() * 3);
            DRmat.setrowsize(dofIdx, sparsityPattern_[spIdx].size());
            DSmat.setrowsize(dofIdx, sparsityPattern_[spIdx].size());
        }
        DDmat.endrowsizes();
        DRmat.endrowsizes();
        DSmat.endrowsizes();

        for (std::size_t dofIdx = 0; dofIdx < DDmat_->rows(); ++dofIdx) {
            const auto spIdx = dofIdx / 3;
            auto nIt = sparsityPattern_[spIdx].begin();
            for (auto nEndIt = sparsityPattern_[spIdx].end(); nIt != nEndIt; ++nIt) {
                DDmat.addindex(dofIdx, *nIt * 3);
                DDmat.addindex(dofIdx, *nIt * 3 + 1);
                DDmat.addindex(dofIdx, *nIt * 3 + 2);

                DRmat.addindex(dofIdx, *nIt);

                DSmat.addindex(dofIdx, *nIt);
            }
        }
        DDmat.endindices();
        DRmat.endindices();
        DSmat.endindices();
    }

    void reserveRDmatSDmat()
    {
        RDmat_->resetIstlMatrix();
        SDmat_->resetIstlMatrix();
        auto& RDmat = RDmat_->istlMatrix();
        auto& SDmat = SDmat_->istlMatrix();

        assert(RDmat_->rows() == sparsityPattern_.size());
        assert(SDmat_->rows() == sparsityPattern_.size());

        for (std::size_t dofIdx = 0; dofIdx < RDmat_->rows(); ++dofIdx) {
            RDmat.setrowsize(dofIdx, sparsityPattern_[dofIdx].size() * 3);
            SDmat.setrowsize(dofIdx, sparsityPattern_[dofIdx].size() * 3);
        }
        RDmat.endrowsizes();
        SDmat.endrowsizes();

        for (std::size_t dofIdx = 0; dofIdx < RDmat_->rows(); ++dofIdx) {
            auto nIt = sparsityPattern_[dofIdx].begin();
            for (auto nEndIt = sparsityPattern_[dofIdx].end(); nIt != nEndIt; ++nIt) {
                RDmat.addindex(dofIdx, *nIt * 3);
                RDmat.addindex(dofIdx, *nIt * 3 + 1);
                RDmat.addindex(dofIdx, *nIt * 3 + 2);

                SDmat.addindex(dofIdx, *nIt * 3);
                SDmat.addindex(dofIdx, *nIt * 3 + 1);
                SDmat.addindex(dofIdx, *nIt * 3 + 2);
            }
        }
        RDmat.endindices();
        SDmat.endindices();
    }

    void initializeSubResiduals_()
    {
        if (parentRes_ == nullptr) {
            return;
        }

        // Resize residuals
        DRes_.resize(parentRes_->size() * 3);
        RRes_.resize(parentRes_->size());
        SRes_.resize(parentRes_->size());
    }

    const Matrix* parentMat_ = nullptr;
    const Vector* parentRes_ = nullptr;

    std::vector<std::set<unsigned> > sparsityPattern_;

    std::unique_ptr<DispDispMatrixT<Scalar> > DDmat_{};
    std::unique_ptr<DispRotMatrixT<Scalar> > DRmat_{};
    std::unique_ptr<DispSPresMatrixT<Scalar> > DSmat_{};

    std::unique_ptr<RotDispMatrixT<Scalar> > RDmat_{};
    std::unique_ptr<RotRotMatrixT<Scalar> > RRmat_{};
    std::unique_ptr<RotSPresMatrixT<Scalar> > RSmat_{};

    std::unique_ptr<SPresDispMatrixT<Scalar> > SDmat_{};
    std::unique_ptr<SPresRotMatrixT<Scalar> > SRmat_{};
    std::unique_ptr<SPresSPresMatrixT<Scalar> > SSmat_{};

    DispVectorT<Scalar> DRes_{};
    RotVectorT<Scalar> RRes_{};
    SPresVectorT<Scalar> SRes_{};
};
}

#endif //MATRIX_RESIDUAL_SPLITTER_HYPRE_HPP
