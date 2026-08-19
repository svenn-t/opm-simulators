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
#ifndef OPM_TPSA_MATRIX_HPP
#define OPM_TPSA_MATRIX_HPP

#include <opm/common/ErrorMacros.hpp>
#include <opm/simulators/linalg/matrixblock.hh>
#include <opm/simulators/linalg/tpsa/TpsaTypes.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <vector>

namespace Opm::Linear
{

template <class Scalar>
class TpsaMatrix;

/*!
 * \brief Handle on one block of the TPSA Jacobian.
 *
 * Replaces the `MatrixBlock*` that IstlSparseMatrixAdapter::blockAddress()
 * returns.  The Jacobian is not stored as dense 7x7 blocks but scattered over
 * 19 sub-matrices (see TpsaTypes.hpp); this class holds the flat index of the
 * block within the shared sparsity pattern and scatters a dense 7x7
 * contribution into the sub-matrices on `+=`.
 *
 * `operator*` returns the object itself so that the linearizer's
 * `*blockAddress += localBlock` syntax works unchanged for both this class and
 * a plain `MatrixBlock*`.
 */
template <class Scalar>
class TpsaBlockRef
{
public:
    using MatrixBlock = Opm::MatrixBlock<Scalar, numTpsaEq, numTpsaEq>;

    TpsaBlockRef() = default;

    TpsaBlockRef(const TpsaMatrix<Scalar>& matrix, std::size_t flatIdx)
        : matrix_(&matrix)
          , k_(flatIdx)
    {
    }

    // The write-through operations below are const, just as writing through a
    // `MatrixBlock* const` is: they modify the matrix, not the handle.
    const TpsaBlockRef& operator*() const
    {
        return *this;
    }

    //! \brief Scatter a dense 7x7 contribution into the sub-matrices.
    const TpsaBlockRef& operator+=(const MatrixBlock& b) const
    {
        apply_([](Scalar& stored, const Scalar& dense) {
                   stored += dense;
               },
               b);

        return *this;
    }

    //! \brief Overwrite the stored entries with a dense 7x7 block.
    const TpsaBlockRef& operator=(const MatrixBlock& b) const
    {
        apply_([](Scalar& stored, const Scalar& dense) {
                   stored = dense;
               },
               b);

        return *this;
    }

    //! \brief Set every stored entry of this block to a scalar.
    const TpsaBlockRef& operator=(Scalar value) const
    {
        const MatrixBlock b(value);

        return *this = b;
    }

    /*!
     * \brief Gather the stored entries into a dense 7x7 block.
     *
     * The six displacement-displacement off-diagonal entries are not stored and
     * come back as zero.
     */
    void gather(MatrixBlock& b) const
    {
        b = Scalar(0.0);
        apply_([](Scalar& stored, Scalar& dense) {
                   dense = stored;
               },
               b);
    }

private:
    // Templated on the block type so that the same index map serves both the
    // scattering (const block) and the gathering (mutable block) direction.
    template <class Op, class Block>
    void apply_(Op op, Block& b) const;

    const TpsaMatrix<Scalar>* matrix_{nullptr};
    std::size_t k_{0};
};

/*!
 * \brief The TPSA Jacobian, stored field-split.
 *
 * Drop-in replacement for Linear::IstlSparseMatrixAdapter as the TPSA
 * SparseMatrixAdapter property.  The linearizer still hands it dense 7x7
 * blocks; internally the entries go straight into the 19 sub-matrices the block
 * preconditioner and Hypre need, so no splitting pass is required between
 * assembly and the linear solve.
 *
 * All sub-matrices are built from the same sparsity pattern, so the k-th
 * nonzero block occupies the same position in each of them.  That is what makes
 * a 16-byte TpsaBlockRef sufficient, and it is why the values of every
 * sub-matrix can be addressed as one contiguous array (the same assumption
 * Hypre's transfer layer makes when it passes `&A[0][0][0][0]` to
 * HYPRE_IJMatrixSetValues2).
 */
template <class Scalar>
class TpsaMatrix
{
    friend class TpsaBlockRef<Scalar>;

public:
    //! \brief What the linear solver operates on.
    using IstlMatrix = TpsaMatrixView<Scalar>;

    //! \brief Dense local block the linearizer accumulates into.
    using MatrixBlock = Opm::MatrixBlock<Scalar, numTpsaEq, numTpsaEq>;

    //! \brief What blockAddress() returns.
    using BlockAddress = TpsaBlockRef<Scalar>;

    using field_type = Scalar;

    TpsaMatrix(std::size_t rows, std::size_t columns)
        : rows_(rows)
          , columns_(columns)
    {
    }

    template <class Simulator>
    explicit TpsaMatrix(const Simulator& simulator)
        : TpsaMatrix(simulator.model().numTotalDof(), simulator.model().numTotalDof())
    {
    }

    // The cached value-array base pointers and the sub-matrix view point into
    // this object, so it must not be copied or moved.
    TpsaMatrix(const TpsaMatrix&) = delete;

    TpsaMatrix(TpsaMatrix&&) = delete;

    TpsaMatrix& operator=(const TpsaMatrix&) = delete;

    TpsaMatrix& operator=(TpsaMatrix&&) = delete;

    ~TpsaMatrix() = default;

    /*!
     * \brief Allocate all sub-matrices from a common sparsity pattern.
     */
    template <class Set>
    void reserve(const std::vector<Set>& sparsityPattern)
    {
        if (sparsityPattern.size() != rows_) {
            OPM_THROW(std::logic_error,
                      "TPSA: sparsity pattern does not match the number of matrix rows");
        }

        // Flatten the pattern once; the column indices of a std::set are
        // already ascending, which is the order BCRSMatrix stores them in.
        rowStart_.resize(rows_ + 1);
        rowStart_[0] = 0;
        for (std::size_t row = 0; row < rows_; ++row) {
            rowStart_[row + 1] = rowStart_[row] + sparsityPattern[row].size();
        }
        nnz_ = rowStart_[rows_];

        colIdx_.clear();
        colIdx_.reserve(nnz_);
        for (std::size_t row = 0; row < rows_; ++row) {
            colIdx_.insert(colIdx_.end(),
                           sparsityPattern[row].begin(),
                           sparsityPattern[row].end());
        }

        forEachSubMatrix_([&](auto& subMatrix) {
            reserveSubMatrix_(subMatrix, sparsityPattern);
        });

        // Initialize base_ pointer to sub-matrices and set up TpsaMatrixView
        cacheValueArrays_();
        wireView_();
    }

    /*!
     * \brief Handle on the block at (rowIdx, colIdx).
     *
     * Only called while the sparsity pattern is set up, so the linear scan over
     * the row is not on any hot path.
     */
    BlockAddress blockAddress(const std::size_t rowIdx, const std::size_t colIdx) const
    {
        return BlockAddress(*this, flatIndex_(rowIdx, colIdx));
    }

    //! \brief Set all matrix entries to zero.
    void clear()
    {
        // Return early, if called before reserve()
        if (nnz_ == 0) {
            return;
        }

        // Note: base_ points to start entry of sub-matrix thus suitable to use with std::fill_n
        forEachSubMatrixWithIndex_([&](auto& subMatrix, std::size_t subIdx) {
            std::fill_n(base_[subIdx], nnz_ * blockScalars_(subMatrix), Scalar(0.0));
        });
    }

    /*!
     * \brief Set the given row to zero, except for the main diagonal.
     *
     * The main diagonal of the block on the diagonal is set to \p diag. Written
     * as dense blocks so that the field split is applied by TpsaBlockRef, i.e. by
     * the same index map assembly goes through, rather than by a second
     * description of which slots are field-diagonal.
     */
    void clearRow(const std::size_t row, const Scalar diag = 1.0)
    {
        const MatrixBlock zeroBlock(Scalar(0.0));

        MatrixBlock diagBlock(Scalar(0.0));
        for (int eqIdx = 0; eqIdx < numTpsaEq; ++eqIdx) {
            diagBlock[eqIdx][eqIdx] = diag;
        }

        for (std::size_t k = rowStart_[row]; k < rowStart_[row + 1]; ++k) {
            BlockAddress(*this, k) = (colIdx_[k] == row) ? diagBlock : zeroBlock;
        }
    }

    //! \brief Zero out the overlap rows and put the identity on their diagonal.
    void makeOverlapRowsInvalid(const std::vector<int>& overlapRows)
    {
        for (const int row : overlapRows) {
            clearRow(static_cast<std::size_t>(row), Scalar(1.0));
        }
    }

    //! \brief Fill \p value with the stored entries of the given block.
    void block(const std::size_t rowIdx, const std::size_t colIdx, MatrixBlock& value) const
    {
        blockAddress(rowIdx, colIdx).gather(value);
    }

    void setBlock(const std::size_t rowIdx, const std::size_t colIdx, const MatrixBlock& value)
    {
        blockAddress(rowIdx, colIdx) = value;
    }

    void addToBlock(const std::size_t rowIdx, const std::size_t colIdx, const MatrixBlock& value)
    {
        blockAddress(rowIdx, colIdx) += value;
    }

    //! \brief No local caching, so nothing to commit.
    void commit()
    {
    }

    //! \brief The structure is already solver-ready after reserve().
    void finalize()
    {
    }

    IstlMatrix& istlMatrix()
    {
        return view_;
    }

    const IstlMatrix& istlMatrix() const
    {
        return view_;
    }

    std::size_t rows() const
    {
        return rows_;
    }

    std::size_t cols() const
    {
        return columns_;
    }

    std::size_t N() const
    {
        return rows_;
    }

    std::size_t M() const
    {
        return columns_;
    }

    std::size_t nonzeroes() const
    {
        return nnz_;
    }

    // Sub-matrix accessors.  dd00/dd11/dd22 and spsp are the scalar blocks Hypre
    // can precondition.
    DispDispMatrix00T<Scalar>& dd00()
    {
        return dd00_;
    }

    DispDispMatrix11T<Scalar>& dd11()
    {
        return dd11_;
    }

    DispDispMatrix22T<Scalar>& dd22()
    {
        return dd22_;
    }

    RotRotMatrixT<Scalar>& rr()
    {
        return rr_;
    }

    SPresSPresMatrixT<Scalar>& spsp()
    {
        return spsp_;
    }

    const DispDispMatrix00T<Scalar>& dd00() const
    {
        return dd00_;
    }

    const DispDispMatrix11T<Scalar>& dd11() const
    {
        return dd11_;
    }

    const DispDispMatrix22T<Scalar>& dd22() const
    {
        return dd22_;
    }

    const RotRotMatrixT<Scalar>& rr() const
    {
        return rr_;
    }

    const SPresSPresMatrixT<Scalar>& spsp() const
    {
        return spsp_;
    }

private:
    // Slots in the base_ array.  The order must match the tuple
    // returned by subMatrices_().
    enum SubMatrixIdx : std::size_t
    {
        DD00, DD11, DD22,
        DR0, DR1, DR2,
        DSP0, DSP1, DSP2,
        RD0, RD1, RD2,
        RR, RSP,
        SPD0, SPD1, SPD2,
        SPR, SPSP,
        numSubMatrices
    };

    /*!
     * \brief Scalars per block of a sub-matrix, i.e. the stride of its flat
     *        value array.
     *
     * Read off the sub-matrix' own block type, so there is no parallel table to
     * keep in step with the slot order.
     */
    template <class SubMatrix>
    static constexpr std::size_t blockScalars_(const SubMatrix&)
    {
        using Block = typename SubMatrix::block_type;

        return static_cast<std::size_t>(Block::rows) * static_cast<std::size_t>(Block::cols);
    }

    auto subMatrices_()
    {
        return std::tie(dd00_,
                        dd11_,
                        dd22_,
                        dr0_,
                        dr1_,
                        dr2_,
                        dsp0_,
                        dsp1_,
                        dsp2_,
                        rd0_,
                        rd1_,
                        rd2_,
                        rr_,
                        rsp_,
                        spd0_,
                        spd1_,
                        spd2_,
                        spr_,
                        spsp_);
    }

    template <class Op>
    void forEachSubMatrix_(Op op)
    {
        std::apply([&](auto&... subMatrix) {
                       (op(subMatrix), ...);
                   },
                   subMatrices_());
    }

    template <class Op>
    void forEachSubMatrixWithIndex_(Op op)
    {
        std::size_t subIdx = 0;
        std::apply([&](auto&... subMatrix) {
                       (op(subMatrix, subIdx++), ...);
                   },
                   subMatrices_());
    }

    template <class SubMatrix, class Set>
    void reserveSubMatrix_(SubMatrix& subMatrix, const std::vector<Set>& sparsityPattern)
    {
        subMatrix.setBuildMode(SubMatrix::random);
        subMatrix.setSize(rows_, columns_);

        for (std::size_t row = 0; row < rows_; ++row) {
            subMatrix.setrowsize(row, sparsityPattern[row].size());
        }
        subMatrix.endrowsizes();

        for (std::size_t row = 0; row < rows_; ++row) {
            for (const auto& col : sparsityPattern[row]) {
                subMatrix.addindex(row, col);
            }
        }
        // Note: all entries in subMatrix are Scalar(0.0) by default construction in endindices()
        subMatrix.endindices();
    }

    /*!
     * \brief Cache the base pointer of every sub-matrix' value array.
     *
     * Also verifies that the k-th nonzero really is at `base + k*stride` in
     * every sub-matrix, i.e. that the sub-matrices share both the pattern and
     * the storage order.  Everything else in this class rests on that.
     */
    void cacheValueArrays_()
    {
        forEachSubMatrixWithIndex_([&](auto& subMatrix, std::size_t subIdx) {
            const std::size_t stride = blockScalars_(subMatrix);
            Scalar* base = nullptr;
            std::size_t k = 0;

            for (auto rowIt = subMatrix.begin(); rowIt != subMatrix.end(); ++rowIt) {
                for (auto colIt = rowIt->begin(); colIt != rowIt->end(); ++colIt, ++k) {
                    Scalar* entry = &(*colIt)[0][0];
                    if (k == 0) {
                        base = entry;
                    }

                    if (entry != base + k * stride || colIt.index() != colIdx_[k]) {
                        OPM_THROW(std::logic_error,
                                  "TPSA: sub-matrix storage is not the contiguous, pattern-ordered "
                                  "layout needed in TpsaMatrix");
                    }
                }
            }

            if (k != nnz_) {
                OPM_THROW(std::logic_error,
                          "TPSA: sub-matrix has an unexpected number of nonzeroes");
            }

            base_[subIdx] = base;
        });
    }

    void wireView_()
    {
        view_.M11_00 = &dd00_;
        view_.M12_00 = &dr0_;
        view_.M13_00 = &dsp0_;

        view_.M11_11 = &dd11_;
        view_.M12_10 = &dr1_;
        view_.M13_10 = &dsp1_;

        view_.M11_22 = &dd22_;
        view_.M12_20 = &dr2_;
        view_.M13_20 = &dsp2_;

        view_.M21_00 = &rd0_;
        view_.M21_01 = &rd1_;
        view_.M21_02 = &rd2_;
        view_.M22 = &rr_;
        view_.M23 = &rsp_;

        view_.M31_00 = &spd0_;
        view_.M31_01 = &spd1_;
        view_.M31_02 = &spd2_;
        view_.M32 = &spr_;
        view_.M33 = &spsp_;
    }

    std::size_t flatIndex_(std::size_t rowIdx, std::size_t colIdx) const
    {
        const auto begin = colIdx_.begin() + rowStart_[rowIdx];
        const auto end = colIdx_.begin() + rowStart_[rowIdx + 1];
        const auto it = std::lower_bound(begin, end, static_cast<unsigned>(colIdx));
        if (it == end || *it != static_cast<unsigned>(colIdx)) {
            OPM_THROW(std::logic_error,
                      "TPSA: requested a matrix block outside the sparsity pattern");
        }

        return rowStart_[rowIdx] + static_cast<std::size_t>(it - begin);
    }

    std::size_t rows_{0};
    std::size_t columns_{0};
    std::size_t nnz_{0};

    // Flattened sparsity pattern, shared by all sub-matrices.
    std::vector<std::size_t> rowStart_{};
    std::vector<unsigned> colIdx_{};

    // Base pointers into the sub-matrices' contiguous value arrays.
    std::array<Scalar*, numSubMatrices> base_{};

    DispDispMatrix00T<Scalar> dd00_{};
    DispDispMatrix11T<Scalar> dd11_{};
    DispDispMatrix22T<Scalar> dd22_{};

    DispRotMatrix0T<Scalar> dr0_{};
    DispRotMatrix1T<Scalar> dr1_{};
    DispRotMatrix2T<Scalar> dr2_{};

    DispSPresMatrix0T<Scalar> dsp0_{};
    DispSPresMatrix1T<Scalar> dsp1_{};
    DispSPresMatrix2T<Scalar> dsp2_{};

    RotDispMatrix0T<Scalar> rd0_{};
    RotDispMatrix1T<Scalar> rd1_{};
    RotDispMatrix2T<Scalar> rd2_{};

    RotRotMatrixT<Scalar> rr_{};
    RotSPresMatrixT<Scalar> rsp_{};

    SPresDispMatrix0T<Scalar> spd0_{};
    SPresDispMatrix1T<Scalar> spd1_{};
    SPresDispMatrix2T<Scalar> spd2_{};

    SPresRotMatrixT<Scalar> spr_{};

    SPresSPresMatrixT<Scalar> spsp_{};

    IstlMatrix view_{};
};

// ---------------------------------------------------------------------------
// TpsaBlockRef implementation (needs the complete TpsaMatrix).
//
// The index map below is the field split of a dense 7x7 block:
//   rows/cols 0,1,2 -> u_x, u_y, u_z   rows/cols 3,4,5 -> rot   row/col 6 -> p_s
// The six displacement-displacement off-diagonal entries b[0][1], b[0][2],
// b[1][0], b[1][2], b[2][0], b[2][1] are deliberately not stored.
// ---------------------------------------------------------------------------
template <class Scalar>
template <class Op, class Block>
void
TpsaBlockRef<Scalar>::apply_(Op op, Block& b) const
{
    using Matrix = TpsaMatrix<Scalar>;
    const auto& base = matrix_->base_;
    const std::size_t k = k_;

    // Displacement-displacement (diagonal components only)
    op(base[Matrix::DD00][k], b[0][0]);
    op(base[Matrix::DD11][k], b[1][1]);
    op(base[Matrix::DD22][k], b[2][2]);

    // Displacement-rotation: three 1x3 blocks
    for (int j = 0; j < numRotDofs; ++j) {
        op(base[Matrix::DR0][3 * k + j], b[0][3 + j]);
        op(base[Matrix::DR1][3 * k + j], b[1][3 + j]);
        op(base[Matrix::DR2][3 * k + j], b[2][3 + j]);
    }

    // Displacement-solid pressure: three 1x1 blocks
    op(base[Matrix::DSP0][k], b[0][6]);
    op(base[Matrix::DSP1][k], b[1][6]);
    op(base[Matrix::DSP2][k], b[2][6]);

    // Rotation-displacement: three 3x1 blocks
    for (int i = 0; i < numRotDofs; ++i) {
        op(base[Matrix::RD0][3 * k + i], b[3 + i][0]);
        op(base[Matrix::RD1][3 * k + i], b[3 + i][1]);
        op(base[Matrix::RD2][3 * k + i], b[3 + i][2]);
    }

    // Rotation-rotation: one 3x3 block
    for (int i = 0; i < numRotDofs; ++i) {
        for (int j = 0; j < numRotDofs; ++j) {
            op(base[Matrix::RR][9 * k + 3 * i + j], b[3 + i][3 + j]);
        }
    }

    // Rotation-solid pressure: one 3x1 block
    for (int i = 0; i < numRotDofs; ++i) {
        op(base[Matrix::RSP][3 * k + i], b[3 + i][6]);
    }

    // Solid pressure-displacement: three 1x1 blocks
    op(base[Matrix::SPD0][k], b[6][0]);
    op(base[Matrix::SPD1][k], b[6][1]);
    op(base[Matrix::SPD2][k], b[6][2]);

    // Solid pressure-rotation: one 1x3 block
    for (int j = 0; j < numRotDofs; ++j) {
        op(base[Matrix::SPR][3 * k + j], b[6][3 + j]);
    }

    // Solid pressure-solid pressure: one 1x1 block
    op(base[Matrix::SPSP][k], b[6][6]);
}

} // namespace Opm::Linear

#endif // OPM_TPSA_MATRIX_HPP
