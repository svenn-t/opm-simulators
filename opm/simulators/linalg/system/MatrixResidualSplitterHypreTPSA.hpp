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
        assignSubMatrices(*DDmat00_,
                          *DDmat11_,
                          *DDmat22_,
                          *DRmat0_,
                          *DRmat1_,
                          *DRmat2_,
                          *DSmat0_,
                          *DSmat1_,
                          *DSmat2_,
                          *RDmat0_,
                          *RDmat1_,
                          *RDmat2_,
                          *RRmat_,
                          *RSmat_,
                          *SDmat0_,
                          *SDmat1_,
                          *SDmat2_,
                          *SRmat_,
                          *SSmat_);

    }

    void assignSubMatrices(DispDispMatrix00T<Scalar>& DDmat00,
                           DispDispMatrix11T<Scalar>& DDmat11,
                           DispDispMatrix22T<Scalar>& DDmat22,
                           DispRotMatrix0T<Scalar>& DRmat0,
                           DispRotMatrix1T<Scalar>& DRmat1,
                           DispRotMatrix2T<Scalar>& DRmat2,
                           DispSPresMatrix0T<Scalar>& DSmat0,
                           DispSPresMatrix1T<Scalar>& DSmat1,
                           DispSPresMatrix2T<Scalar>& DSmat2,
                           RotDispMatrix0T<Scalar>& RDmat0,
                           RotDispMatrix1T<Scalar>& RDmat1,
                           RotDispMatrix2T<Scalar>& RDmat2,
                           RotRotMatrixT<Scalar>& RRmat,
                           RotSPresMatrixT<Scalar>& RSmat,
                           SPresDispMatrix0T<Scalar>& SDmat0,
                           SPresDispMatrix1T<Scalar>& SDmat1,
                           SPresDispMatrix2T<Scalar>& SDmat2,
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
                // Displacement-displacement matrix
                auto* DDMatBlock00 = DDmat00.blockAddress(row.index(), col.index());
                auto* DDMatBlock11 = DDmat11.blockAddress(row.index(), col.index());
                auto* DDMatBlock22 = DDmat22.blockAddress(row.index(), col.index());
                (*DDMatBlock00)[0][0] = parentBlock[0][0];
                (*DDMatBlock11)[0][0] = parentBlock[1][1];
                (*DDMatBlock22)[0][0] = parentBlock[2][2];

                // Displacement-rotation matrix
                auto* DRMatBlock0 = DRmat0.blockAddress(row.index(), col.index());
                auto* DRMatBlock1 = DRmat1.blockAddress(row.index(), col.index());
                auto* DRMatBlock2 = DRmat2.blockAddress(row.index(), col.index());
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
                auto* DSMatBlock0 = DSmat0.blockAddress(row.index(), col.index());
                auto* DSMatBlock1 = DSmat1.blockAddress(row.index(), col.index());
                auto* DSMatBlock2 = DSmat2.blockAddress(row.index(), col.index());
                (*DSMatBlock0)[0][0] = parentBlock[0][6];
                (*DSMatBlock1)[0][0] = parentBlock[1][6];
                (*DSMatBlock2)[0][0] = parentBlock[2][6];

                // Rotation-displacement matrix
                auto* RDMatBlock0 = RDmat0.blockAddress(row.index(), col.index());
                auto* RDMatBlock1 = RDmat1.blockAddress(row.index(), col.index());
                auto* RDMatBlock2 = RDmat2.blockAddress(row.index(), col.index());
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
                auto* SDMatBlock0 = SDmat0.blockAddress(row.index(), col.index());
                auto* SDMatBlock1 = SDmat1.blockAddress(row.index(), col.index());
                auto* SDMatBlock2 = SDmat2.blockAddress(row.index(), col.index());
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
        // Initialize and assign sub-residuals
        initializeSubResiduals_();
        assignSubResiduals(DRes0_, DRes1_, DRes2_, RRes_, SRes_);
    }

    void assignSubResiduals(DispVector0T<Scalar>& DRes0,
                            DispVector1T<Scalar>& DRes1,
                            DispVector2T<Scalar>& DRes2,
                            RotVectorT<Scalar>& RRes,
                            SPresVectorT<Scalar>& SRes)
    {
        if (parentRes_ == nullptr) {
            return;
        }

        // Assign sub-residuals
        for (std::size_t i = 0; i < parentRes_->size(); ++i) {
            DRes0[i][0] = (*parentRes_)[i][0];
            DRes1[i][0] = (*parentRes_)[i][1];
            DRes2[i][0] = (*parentRes_)[i][2];

            RRes[i][0] = (*parentRes_)[i][3];
            RRes[i][1] = (*parentRes_)[i][4];
            RRes[i][2] = (*parentRes_)[i][5];

            SRes[i][0] = (*parentRes_)[i][6];
        }
    }

    //
    // Transfer ownership
    //
    std::unique_ptr<DispDispMatrix00T<Scalar> > takeDispDisp00Matrix() noexcept
    {
        return std::move(DDmat00_);
    }

    std::unique_ptr<DispDispMatrix11T<Scalar> > takeDispDisp11Matrix() noexcept
    {
        return std::move(DDmat11_);
    }

    std::unique_ptr<DispDispMatrix22T<Scalar> > takeDispDisp22Matrix() noexcept
    {
        return std::move(DDmat22_);
    }

    std::unique_ptr<DispRotMatrix0T<Scalar> > takeDispRot0Matrix() noexcept
    {
        return std::move(DRmat0_);
    }

    std::unique_ptr<DispRotMatrix1T<Scalar> > takeDispRot1Matrix() noexcept
    {
        return std::move(DRmat1_);
    }

    std::unique_ptr<DispRotMatrix2T<Scalar> > takeDispRot2Matrix() noexcept
    {
        return std::move(DRmat2_);
    }

    std::unique_ptr<DispSPresMatrix0T<Scalar> > takeDispSPres0Matrix() noexcept
    {
        return std::move(DSmat0_);
    }

    std::unique_ptr<DispSPresMatrix1T<Scalar> > takeDispSPres1Matrix() noexcept
    {
        return std::move(DSmat1_);
    }

    std::unique_ptr<DispSPresMatrix2T<Scalar> > takeDispSPres2Matrix() noexcept
    {
        return std::move(DSmat2_);
    }

    std::unique_ptr<RotDispMatrix0T<Scalar> > takeRotDisp0Matrix() noexcept
    {
        return std::move(RDmat0_);
    }

    std::unique_ptr<RotDispMatrix1T<Scalar> > takeRotDisp1Matrix() noexcept
    {
        return std::move(RDmat1_);
    }

    std::unique_ptr<RotDispMatrix2T<Scalar> > takeRotDisp2Matrix() noexcept
    {
        return std::move(RDmat2_);
    }

    std::unique_ptr<RotRotMatrixT<Scalar> > takeRotRotMatrix() noexcept
    {
        return std::move(RRmat_);
    }

    std::unique_ptr<RotSPresMatrixT<Scalar> > takeRotSPresMatrix() noexcept
    {
        return std::move(RSmat_);
    }

    std::unique_ptr<SPresDispMatrix0T<Scalar> > takeSPresDisp0Matrix() noexcept
    {
        return std::move(SDmat0_);
    }

    std::unique_ptr<SPresDispMatrix1T<Scalar> > takeSPresDisp1Matrix() noexcept
    {
        return std::move(SDmat1_);
    }

    std::unique_ptr<SPresDispMatrix2T<Scalar> > takeSPresDisp2Matrix() noexcept
    {
        return std::move(SDmat2_);
    }

    std::unique_ptr<SPresRotMatrixT<Scalar> > takeSPresRotMatrix() noexcept
    {
        return std::move(SRmat_);
    }

    std::unique_ptr<SPresSPresMatrixT<Scalar> > takeSPresSPresMatrix() noexcept
    {
        return std::move(SSmat_);
    }

    DispVector0T<Scalar> takeDisp0Vector() noexcept
    {
        return std::move(DRes0_);
    }

    DispVector1T<Scalar> takeDisp1Vector() noexcept
    {
        return std::move(DRes1_);
    }

    DispVector2T<Scalar> takeDisp2Vector() noexcept
    {
        return std::move(DRes2_);
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
    DispDispMatrix00T<Scalar>& dispDisp00Matrix()
    {
        return *DDmat00_;
    }

    const DispDispMatrix00T<Scalar>& dispDisp00Matrix() const
    {
        return *DDmat00_;
    }

    DispDispMatrix11T<Scalar>& dispDisp11Matrix()
    {
        return *DDmat11_;
    }

    const DispDispMatrix11T<Scalar>& dispDisp11Matrix() const
    {
        return *DDmat11_;
    }

    DispDispMatrix22T<Scalar>& dispDisp22Matrix()
    {
        return *DDmat22_;
    }

    const DispDispMatrix22T<Scalar>& dispDisp22Matrix() const
    {
        return *DDmat22_;
    }

    DispRotMatrix0T<Scalar>& dispRot0Matrix()
    {
        return *DRmat0_;
    }

    const DispRotMatrix0T<Scalar>& dispRot0Matrix() const
    {
        return *DRmat0_;
    }

    DispRotMatrix1T<Scalar>& dispRot1Matrix()
    {
        return *DRmat1_;
    }

    const DispRotMatrix1T<Scalar>& dispRot1Matrix() const
    {
        return *DRmat1_;
    }

    DispRotMatrix2T<Scalar>& dispRot2Matrix()
    {
        return *DRmat2_;
    }

    const DispRotMatrix2T<Scalar>& dispRot2Matrix() const
    {
        return *DRmat2_;
    }

    DispSPresMatrix0T<Scalar>& dispSPres0Matrix()
    {
        return *DSmat0_;
    }

    const DispSPresMatrix0T<Scalar>& dispSPres0Matrix() const
    {
        return *DSmat0_;
    }

    DispSPresMatrix1T<Scalar>& dispSPres1Matrix()
    {
        return *DSmat1_;
    }

    const DispSPresMatrix1T<Scalar>& dispSPres1Matrix() const
    {
        return *DSmat1_;
    }

    DispSPresMatrix2T<Scalar>& dispSPres2Matrix()
    {
        return *DSmat2_;
    }

    const DispSPresMatrix2T<Scalar>& dispSPres2Matrix() const
    {
        return *DSmat2_;
    }

    RotDispMatrix0T<Scalar>& rotDisp0Matrix()
    {
        return *RDmat0_;
    }

    const RotDispMatrix0T<Scalar>& rotDisp0Matrix() const
    {
        return *RDmat0_;
    }

    RotDispMatrix1T<Scalar>& rotDisp1Matrix()
    {
        return *RDmat1_;
    }

    const RotDispMatrix1T<Scalar>& rotDisp1Matrix() const
    {
        return *RDmat1_;
    }

    RotDispMatrix2T<Scalar>& rotDisp2Matrix()
    {
        return *RDmat2_;
    }

    const RotDispMatrix2T<Scalar>& rotDisp2Matrix() const
    {
        return *RDmat2_;
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

    SPresDispMatrix0T<Scalar>& sPresDisp0Matrix()
    {
        return *SDmat0_;
    }

    const SPresDispMatrix0T<Scalar>& sPresDisp0Matrix() const
    {
        return *SDmat0_;
    }

    SPresDispMatrix1T<Scalar>& sPresDisp1Matrix()
    {
        return *SDmat1_;
    }

    const SPresDispMatrix1T<Scalar>& sPresDisp1Matrix() const
    {
        return *SDmat1_;
    }

    SPresDispMatrix2T<Scalar>& sPresDisp2Matrix()
    {
        return *SDmat2_;
    }

    const SPresDispMatrix2T<Scalar>& sPresDisp2Matrix() const
    {
        return *SDmat2_;
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

    const DispVector0T<Scalar>& disp0Vector() const
    {
        return DRes0_;
    }

    const DispVector1T<Scalar>& disp1Vector() const
    {
        return DRes1_;
    }

    const DispVector2T<Scalar>& disp2Vector() const
    {
        return DRes2_;
    }

    const RotVectorT<Scalar>& rotVector() const
    {
        return RRes_;
    }

    const SPresVectorT<Scalar>& sPresVector() const
    {
        return SRes_;
    }

private
:
    void initializeSubMatrices_()
    {
        if (parentMat_ == nullptr || sparsityPattern_.empty()) {
            return;
        }

        // Instantiate sub-matrices
        DDmat00_ = std::make_unique<DispDispMatrix00T<Scalar> >(parentMat_->N(),
                                                                parentMat_->M());
        DDmat11_ = std::make_unique<DispDispMatrix11T<Scalar> >(parentMat_->N(),
                                                                parentMat_->M());
        DDmat22_ = std::make_unique<DispDispMatrix22T<Scalar> >(parentMat_->N(),
                                                                parentMat_->M());
        DRmat0_ = std::make_unique<DispRotMatrix0T<Scalar> >(parentMat_->N(),
                                                             parentMat_->M());
        DRmat1_ = std::make_unique<DispRotMatrix1T<Scalar> >(parentMat_->N(),
                                                             parentMat_->M());
        DRmat2_ = std::make_unique<DispRotMatrix2T<Scalar> >(parentMat_->N(),
                                                             parentMat_->M());
        DSmat0_ = std::make_unique<DispSPresMatrix0T<Scalar> >(parentMat_->N(),
                                                               parentMat_->M());
        DSmat1_ = std::make_unique<DispSPresMatrix1T<Scalar> >(parentMat_->N(),
                                                               parentMat_->M());
        DSmat2_ = std::make_unique<DispSPresMatrix2T<Scalar> >(parentMat_->N(),
                                                               parentMat_->M());

        RDmat0_ = std::make_unique<RotDispMatrix0T<Scalar> >(parentMat_->N(),
                                                             parentMat_->M());
        RDmat1_ = std::make_unique<RotDispMatrix1T<Scalar> >(parentMat_->N(),
                                                             parentMat_->M());
        RDmat2_ = std::make_unique<RotDispMatrix2T<Scalar> >(parentMat_->N(),
                                                             parentMat_->M());
        RRmat_ = std::make_unique<RotRotMatrixT<Scalar> >(parentMat_->N(),
                                                          parentMat_->M());
        RSmat_ = std::make_unique<RotSPresMatrixT<Scalar> >(parentMat_->N(),
                                                            parentMat_->M());

        SDmat0_ = std::make_unique<SPresDispMatrix0T<Scalar> >(parentMat_->N(),
                                                               parentMat_->M());
        SDmat1_ = std::make_unique<SPresDispMatrix1T<Scalar> >(parentMat_->N(),
                                                               parentMat_->M());
        SDmat2_ = std::make_unique<SPresDispMatrix2T<Scalar> >(parentMat_->N(),
                                                               parentMat_->M());
        SRmat_ = std::make_unique<SPresRotMatrixT<Scalar> >(parentMat_->N(),
                                                            parentMat_->M());
        SSmat_ = std::make_unique<SPresSPresMatrixT<Scalar> >(parentMat_->N(),
                                                              parentMat_->M());

        // Set sparsity pattern
        DDmat00_->reserve(sparsityPattern_);
        DDmat11_->reserve(sparsityPattern_);
        DDmat22_->reserve(sparsityPattern_);
        DRmat0_->reserve(sparsityPattern_);
        DRmat1_->reserve(sparsityPattern_);
        DRmat2_->reserve(sparsityPattern_);
        DSmat0_->reserve(sparsityPattern_);
        DSmat1_->reserve(sparsityPattern_);
        DSmat2_->reserve(sparsityPattern_);

        RDmat0_->reserve(sparsityPattern_);
        RDmat1_->reserve(sparsityPattern_);
        RDmat2_->reserve(sparsityPattern_);
        RRmat_->reserve(sparsityPattern_);
        RSmat_->reserve(sparsityPattern_);

        SDmat0_->reserve(sparsityPattern_);
        SDmat1_->reserve(sparsityPattern_);
        SDmat2_->reserve(sparsityPattern_);
        SRmat_->reserve(sparsityPattern_);
        SSmat_->reserve(sparsityPattern_);
    }

    void initializeSubResiduals_()
    {
        if (parentRes_ == nullptr) {
            return;
        }

        // Resize residuals
        DRes0_.resize(parentRes_->size());
        DRes1_.resize(parentRes_->size());
        DRes2_.resize(parentRes_->size());
        RRes_.resize(parentRes_->size());
        SRes_.resize(parentRes_->size());
    }

    const Matrix* parentMat_ = nullptr;
    const Vector* parentRes_ = nullptr;

    std::vector<std::set<unsigned> > sparsityPattern_;

    std::unique_ptr<DispDispMatrix00T<Scalar> > DDmat00_{};
    std::unique_ptr<DispDispMatrix11T<Scalar> > DDmat11_{};
    std::unique_ptr<DispDispMatrix22T<Scalar> > DDmat22_{};
    std::unique_ptr<DispRotMatrix0T<Scalar> > DRmat0_{};
    std::unique_ptr<DispRotMatrix1T<Scalar> > DRmat1_{};
    std::unique_ptr<DispRotMatrix2T<Scalar> > DRmat2_{};
    std::unique_ptr<DispSPresMatrix0T<Scalar> > DSmat0_{};
    std::unique_ptr<DispSPresMatrix1T<Scalar> > DSmat1_{};
    std::unique_ptr<DispSPresMatrix2T<Scalar> > DSmat2_{};

    std::unique_ptr<RotDispMatrix0T<Scalar> > RDmat0_{};
    std::unique_ptr<RotDispMatrix1T<Scalar> > RDmat1_{};
    std::unique_ptr<RotDispMatrix2T<Scalar> > RDmat2_{};
    std::unique_ptr<RotRotMatrixT<Scalar> > RRmat_{};
    std::unique_ptr<RotSPresMatrixT<Scalar> > RSmat_{};

    std::unique_ptr<SPresDispMatrix0T<Scalar> > SDmat0_{};
    std::unique_ptr<SPresDispMatrix1T<Scalar> > SDmat1_{};
    std::unique_ptr<SPresDispMatrix2T<Scalar> > SDmat2_{};
    std::unique_ptr<SPresRotMatrixT<Scalar> > SRmat_{};
    std::unique_ptr<SPresSPresMatrixT<Scalar> > SSmat_{};

    DispVector0T<Scalar> DRes0_{};
    DispVector1T<Scalar> DRes1_{};
    DispVector2T<Scalar> DRes2_{};
    RotVectorT<Scalar> RRes_{};
    SPresVectorT<Scalar> SRes_{};
};
}

#endif //MATRIX_RESIDUAL_SPLITTER_HYPRE_HPP
