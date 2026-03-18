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
#ifndef MATRIX_RESIDUAL_SPLITTER_HPP
#define MATRIX_RESIDUAL_SPLITTER_HPP

#include "SystemTypes.hpp"

namespace Opm
{
template<typename Scalar, typename Matrix, typename Vector>
class MatrixResidualSplitterTPSA
{
public:
    MatrixResidualSplitterTPSA() = default;
    MatrixResidualSplitterTPSA(const Matrix& M,
                               const Vector& residual,
                               const std::vector<std::set<unsigned> >& sparsityPattern)
        : parentMat_(&M)
        , parentRes_(&residual)
        , sparsityPattern_(sparsityPattern)
    {
        // Instantiate sub-matrices
        DDmat_ = std::make_unique<DispDispMatrixT<Scalar>>(M.N(), M.M());
        DRmat_ = std::make_unique<DispRotMatrixT<Scalar>>(M.N(), M.M());
        DSmat_ = std::make_unique<DispSPresMatrixT<Scalar>>(M.N(), M.M());

        RDmat_ = std::make_unique<RotDispMatrixT<Scalar>>(M.N(), M.M());
        RRmat_ = std::make_unique<RotRotMatrixT<Scalar>>(M.N(), M.M());
        RSmat_ = std::make_unique<RotSPresMatrixT<Scalar>>(M.N(), M.M());

        SDmat_ = std::make_unique<SPresDispMatrixT<Scalar>>(M.N(), M.M());
        SRmat_ = std::make_unique<SPresRotMatrixT<Scalar>>(M.N(), M.M());
        SSmat_ = std::make_unique<SPresSPresMatrixT<Scalar>>(M.N(), M.M());
    }

    template <class Simulator>
    MatrixResidualSplitterTPSA(const Simulator& simulator,
                               const Matrix& M,
                               const Vector& residual,
                               const std::vector<std::set<unsigned> >& sparsityPattern)
        : parentMat_(&M)
        , parentRes_(&residual)
        , sparsityPattern_(sparsityPattern)
    {
        // Instantiate sub-matrices
        DDmat_ = std::make_unique<DispDispMatrixT<Scalar>>(simulator);
        DRmat_ = std::make_unique<DispRotMatrixT<Scalar>>(simulator);
        DSmat_ = std::make_unique<DispSPresMatrixT<Scalar>>(simulator);

        RDmat_ = std::make_unique<RotDispMatrixT<Scalar>>(simulator);
        RRmat_ = std::make_unique<RotRotMatrixT<Scalar>>(simulator);
        RSmat_ = std::make_unique<RotSPresMatrixT<Scalar>>(simulator);

        SDmat_ = std::make_unique<SPresDispMatrixT<Scalar>>(simulator);
        SRmat_ = std::make_unique<SPresRotMatrixT<Scalar>>(simulator);
        SSmat_ = std::make_unique<SPresSPresMatrixT<Scalar>>(simulator);
    }

    void generateSubmatrices()
    {
        // Jump out if parent matrix is nullptr or if sparsity pattern has not been set
        if (!parentMat_ || sparsityPattern_.empty()) {
            return;
        }

        // Set sparsity pattern
        DDmat_->reserve(sparsityPattern_);
        DRmat_->reserve(sparsityPattern_);
        DSmat_->reserve(sparsityPattern_);

        RDmat_->reserve(sparsityPattern_);
        RRmat_->reserve(sparsityPattern_);
        RSmat_->reserve(sparsityPattern_);

        SDmat_->reserve(sparsityPattern_);
        SRmat_->reserve(sparsityPattern_);
        SSmat_->reserve(sparsityPattern_);

        // Loop over parent matrix and assign entries to correct sub-matrix
        for (auto row = parentMat_->begin(); row != parentMat_->end(); ++row) {
            for (auto col = row->begin(); col != row->end(); ++col) {
                // Get parent block
                const auto& parentBlock = *col;

                // Assign to sub-matrices
                // Displacement-displacement matrix
                auto* DDMatBlock = DDmat_->blockAddress(row.index(), col.index());
                (*DDMatBlock)[0][0] = parentBlock[0][0];
                (*DDMatBlock)[0][1] = parentBlock[0][1];
                (*DDMatBlock)[0][2] = parentBlock[0][2];
                (*DDMatBlock)[1][0] = parentBlock[1][0];
                (*DDMatBlock)[1][1] = parentBlock[1][1];
                (*DDMatBlock)[1][2] = parentBlock[1][2];
                (*DDMatBlock)[2][0] = parentBlock[2][0];
                (*DDMatBlock)[2][1] = parentBlock[2][1];
                (*DDMatBlock)[2][2] = parentBlock[2][2];

                // Displacement-rotation matrix
                auto* DRMatBlock = DRmat_->blockAddress(row.index(), col.index());
                (*DRMatBlock)[0][0] = parentBlock[0][3];
                (*DRMatBlock)[0][1] = parentBlock[0][4];
                (*DRMatBlock)[0][2] = parentBlock[0][5];
                (*DRMatBlock)[1][0] = parentBlock[1][3];
                (*DRMatBlock)[1][1] = parentBlock[1][4];
                (*DRMatBlock)[1][2] = parentBlock[1][5];
                (*DRMatBlock)[2][0] = parentBlock[2][3];
                (*DRMatBlock)[2][1] = parentBlock[2][4];
                (*DRMatBlock)[2][2] = parentBlock[2][5];

                // Displacement-solid pressure matrix
                auto* DSMatBlock = DSmat_->blockAddress(row.index(), col.index());
                (*DSMatBlock)[0][0] = parentBlock[0][6];
                (*DSMatBlock)[1][0] = parentBlock[1][6];
                (*DSMatBlock)[2][0] = parentBlock[2][6];

                // Rotation-displacement matrix
                auto* RDMatBlock = RDmat_->blockAddress(row.index(), col.index());
                (*RDMatBlock)[0][0] = parentBlock[3][0];
                (*RDMatBlock)[0][1] = parentBlock[3][1];
                (*RDMatBlock)[0][2] = parentBlock[3][2];
                (*RDMatBlock)[1][0] = parentBlock[4][0];
                (*RDMatBlock)[1][1] = parentBlock[4][1];
                (*RDMatBlock)[1][2] = parentBlock[4][2];
                (*RDMatBlock)[2][0] = parentBlock[5][0];
                (*RDMatBlock)[2][1] = parentBlock[5][1];
                (*RDMatBlock)[2][2] = parentBlock[5][2];

                // Rotation-rotation matrix
                auto* RRMatBlock = RRmat_->blockAddress(row.index(), col.index());
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
                auto* RSMatBlock = RSmat_->blockAddress(row.index(), col.index());
                (*RSMatBlock)[0][0] = parentBlock[3][6];
                (*RSMatBlock)[1][0] = parentBlock[4][6];
                (*RSMatBlock)[2][0] = parentBlock[5][6];

                // Solid pressure-displacement matrix
                auto* SDMatBlock = SDmat_->blockAddress(row.index(), col.index());
                (*SDMatBlock)[0][0] = parentBlock[6][0];
                (*SDMatBlock)[0][1] = parentBlock[6][1];
                (*SDMatBlock)[0][2] = parentBlock[6][2];

                // Solid pressure-rotation matrix
                auto* SRMatBlock = SRmat_->blockAddress(row.index(), col.index());
                (*SRMatBlock)[0][0] = parentBlock[6][3];
                (*SRMatBlock)[0][1] = parentBlock[6][4];
                (*SRMatBlock)[0][2] = parentBlock[6][5];

                // Solid pressure-solid pressure matrix
                auto* SSMatBlock = SSmat_->blockAddress(row.index(), col.index());
                (*SSMatBlock)[0][0] = parentBlock[6][6];
            }
        }
    }

    void generateSubResiduals()
    {
        if (!parentRes_) {
            return;
        }

        // Resize residuals
        DRes_.resize(parentRes_->size());
        RRes_.resize(parentRes_->size());
        SRes_.resize(parentRes_->size());

        // Assign sub-residuals
        for (std::size_t i = 0; i < parentRes_->size(); ++i) {
            DRes_[i][0] = (*parentRes_)[i][0];
            DRes_[i][1] = (*parentRes_)[i][1];
            DRes_[i][2] = (*parentRes_)[i][2];

            RRes_[i][0] = (*parentRes_)[i][3];
            RRes_[i][1] = (*parentRes_)[i][4];
            RRes_[i][2] = (*parentRes_)[i][5];

            SRes_[i][0] = (*parentRes_)[i][6];
        }
    }

    //
    // Transfer ownership
    //
    std::unique_ptr<DispDispMatrixT<Scalar>> takeDispDispMatrix() noexcept { return std::move(DDmat_); }
    std::unique_ptr<DispRotMatrixT<Scalar>> takeDispRotMatrix() noexcept { return std::move(DRmat_); }
    std::unique_ptr<DispSPresMatrixT<Scalar>> takeDispSPresMatrix() noexcept { return std::move(DSmat_); }

    std::unique_ptr<RotDispMatrixT<Scalar>> takeRotDispMatrix() noexcept { return std::move(RDmat_); }
    std::unique_ptr<RotRotMatrixT<Scalar>> takeRotRotMatrix() noexcept { return std::move(RRmat_); }
    std::unique_ptr<RotSPresMatrixT<Scalar>> takeRotSPresMatrix() noexcept { return std::move(RSmat_); }

    std::unique_ptr<SPresDispMatrixT<Scalar>> takeSPresDispMatrix()  noexcept { return std::move(SDmat_); }
    std::unique_ptr<SPresRotMatrixT<Scalar>> takeSPresRotMatrix() noexcept { return std::move(SRmat_); }
    std::unique_ptr<SPresSPresMatrixT<Scalar>> takeSPresSPresMatrix() noexcept { return std::move(SSmat_); }

    DispVectorT<Scalar> takeDispVector() noexcept { return std::move(DRes_); }
    RotVectorT<Scalar> takeRotVector() noexcept { return std::move(RRes_); }
    SPresVectorT<Scalar> takeSPresVector() noexcept { return std::move(SRes_); }

    //
    //  Viewers
    //
    DispDispMatrixT<Scalar>& dispDispMatrix() { return *DDmat_; }
    const DispDispMatrixT<Scalar>& dispDispMatrix() const { return *DDmat_; }

    DispRotMatrixT<Scalar>& dispRotMatrix() { return *DRmat_; }
    const DispRotMatrixT<Scalar>& dispRotMatrix() const { return *DRmat_; }

    DispSPresMatrixT<Scalar>& dispSPresMatrix() { return *DSmat_; }
    const DispSPresMatrixT<Scalar>& dispSPresMatrix() const { return *DSmat_; }

    RotDispMatrixT<Scalar>& rotDispMatrix() { return *RDmat_; }
    const RotDispMatrixT<Scalar>& rotDispMatrix() const { return *RDmat_; }

    RotRotMatrixT<Scalar>& rotRotMatrix() { return *RRmat_; }
    const RotRotMatrixT<Scalar>& rotRotMatrix() const { return *RRmat_; }

    RotSPresMatrixT<Scalar>& rotSPresMatrix() { return *RSmat_; }
    const RotSPresMatrixT<Scalar>& rotSPresMatrix() const { return *RSmat_; }

    SPresDispMatrixT<Scalar>& sPresDispMatrix() { return *SDmat_; }
    const SPresDispMatrixT<Scalar>& sPresDispMatrix() const { return *SDmat_; }

    SPresRotMatrixT<Scalar>& sPresRotMatrix() { return *SRmat_; }
    const SPresRotMatrixT<Scalar>& sPresRotMatrix() const { return *SRmat_; }

    SPresSPresMatrixT<Scalar>& sPresSPresMatrix() { return *SSmat_; }
    const SPresSPresMatrixT<Scalar>& sPresSPresMatrix() const { return *SSmat_; }

    const DispVectorT<Scalar>& dispVector() const { return DRes_; }
    const RotVectorT<Scalar>& rotVector() const { return RRes_; }
    const SPresVectorT<Scalar>& sPresVector() const { return SRes_; }

private:
    const Matrix* parentMat_ = nullptr;
    const Vector* parentRes_ = nullptr;

    std::vector<std::set<unsigned> > sparsityPattern_;

    std::unique_ptr<DispDispMatrixT<Scalar>> DDmat_{};
    std::unique_ptr<DispRotMatrixT<Scalar>> DRmat_{};
    std::unique_ptr<DispSPresMatrixT<Scalar>> DSmat_{};

    std::unique_ptr<RotDispMatrixT<Scalar>> RDmat_{};
    std::unique_ptr<RotRotMatrixT<Scalar>> RRmat_{};
    std::unique_ptr<RotSPresMatrixT<Scalar>> RSmat_{};

    std::unique_ptr<SPresDispMatrixT<Scalar>> SDmat_{};
    std::unique_ptr<SPresRotMatrixT<Scalar>> SRmat_{};
    std::unique_ptr<SPresSPresMatrixT<Scalar>> SSmat_{};

    DispVectorT<Scalar> DRes_{};
    RotVectorT<Scalar> RRes_{};
    SPresVectorT<Scalar> SRes_{};
};
}

#endif //MATRIX_RESIDUAL_SPLITTER_HPP
