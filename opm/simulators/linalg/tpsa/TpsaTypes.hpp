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
#ifndef OPM_TPSA_TYPES_HPP
#define OPM_TPSA_TYPES_HPP

#include <opm/simulators/linalg/matrixblock.hh>

#include <dune/common/fvector.hh>
#include <dune/common/indices.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/multitypeblockvector.hh>

#include <cstddef>

/*!
 * \file
 *
 * \brief Field-split representation of the TPSA (elasticity) linear system.
 *
 * The TPSA model has seven equations per cell: three displacement components,
 * three rotation components and one solid pressure.  Instead of storing them as
 * one monolithic matrix of dense 7x7 blocks, the system is split into five
 * fields
 *
 *     field 0: u_x            (1 dof)
 *     field 1: u_y            (1 dof)
 *     field 2: u_z            (1 dof)
 *     field 3: rot_x/y/z      (3 dofs)
 *     field 4: p_solid        (1 dof)
 *
 * so that the diagonal blocks of fields 0, 1, 2 and 4 are genuine scalar
 * (1x1-block) BCRSMatrices.  This is what Hypre's BoomerAMG requires: it is only
 * registered for matrices with `block_type::rows == cols == 1` (see
 * StandardPreconditioners_serial.hpp / StandardPreconditioners_mpi.hpp) and the
 * transfer layer hands `&A[0][0][0][0]` straight to HYPRE_IJMatrixSetValues2.
 *
 * The displacement-displacement off-diagonal couplings (u_x-u_y, u_x-u_z, ...)
 * are deliberately not represented: only 19 of the 25 field couplings exist.
 * The row proxies below therefore only provide `operator[]` overloads for the
 * couplings that are stored, so a reference to a dropped coupling is a compile
 * error rather than a silent zero.
 */

namespace Opm::Linear {

//! \brief Number of dofs of each field.
inline constexpr int numDispDofs = 1;
inline constexpr int numRotDofs = 3;
inline constexpr int numSolidPresDofs = 1;

//! \brief Number of fields the TPSA system is split into.
inline constexpr int numTpsaFields = 5;

//! \brief Total number of TPSA equations per cell.
inline constexpr int numTpsaEq = 3 * numDispDofs + numRotDofs + numSolidPresDofs;

// ---------------------------------------------------------------------------
// Sub-matrix types.  All of them share the sparsity pattern of the TPSA
// stencil; only the block shape differs.
// ---------------------------------------------------------------------------

// Diagonal blocks
template <typename Scalar>
using DispDispMatrix00T = Dune::BCRSMatrix<MatrixBlock<Scalar, numDispDofs, numDispDofs>>;
template <typename Scalar>
using DispDispMatrix11T = Dune::BCRSMatrix<MatrixBlock<Scalar, numDispDofs, numDispDofs>>;
template <typename Scalar>
using DispDispMatrix22T = Dune::BCRSMatrix<MatrixBlock<Scalar, numDispDofs, numDispDofs>>;
template <typename Scalar>
using RotRotMatrixT = Dune::BCRSMatrix<MatrixBlock<Scalar, numRotDofs, numRotDofs>>;
template <typename Scalar>
using SPresSPresMatrixT = Dune::BCRSMatrix<MatrixBlock<Scalar, numSolidPresDofs, numSolidPresDofs>>;

// Displacement-rotation and displacement-solid pressure
template <typename Scalar>
using DispRotMatrix0T = Dune::BCRSMatrix<MatrixBlock<Scalar, numDispDofs, numRotDofs>>;
template <typename Scalar>
using DispRotMatrix1T = DispRotMatrix0T<Scalar>;
template <typename Scalar>
using DispRotMatrix2T = DispRotMatrix0T<Scalar>;
template <typename Scalar>
using DispSPresMatrix0T = Dune::BCRSMatrix<MatrixBlock<Scalar, numDispDofs, numSolidPresDofs>>;
template <typename Scalar>
using DispSPresMatrix1T = DispSPresMatrix0T<Scalar>;
template <typename Scalar>
using DispSPresMatrix2T = DispSPresMatrix0T<Scalar>;

// Rotation-displacement and rotation-solid pressure
template <typename Scalar>
using RotDispMatrix0T = Dune::BCRSMatrix<MatrixBlock<Scalar, numRotDofs, numDispDofs>>;
template <typename Scalar>
using RotDispMatrix1T = RotDispMatrix0T<Scalar>;
template <typename Scalar>
using RotDispMatrix2T = RotDispMatrix0T<Scalar>;
template <typename Scalar>
using RotSPresMatrixT = Dune::BCRSMatrix<MatrixBlock<Scalar, numRotDofs, numSolidPresDofs>>;

// Solid pressure-displacement and solid pressure-rotation
template <typename Scalar>
using SPresDispMatrix0T = Dune::BCRSMatrix<MatrixBlock<Scalar, numSolidPresDofs, numDispDofs>>;
template <typename Scalar>
using SPresDispMatrix1T = SPresDispMatrix0T<Scalar>;
template <typename Scalar>
using SPresDispMatrix2T = SPresDispMatrix0T<Scalar>;
template <typename Scalar>
using SPresRotMatrixT = Dune::BCRSMatrix<MatrixBlock<Scalar, numSolidPresDofs, numRotDofs>>;

// ---------------------------------------------------------------------------
// Sub-vector types
// ---------------------------------------------------------------------------
template <typename Scalar>
using DispVector0T = Dune::BlockVector<Dune::FieldVector<Scalar, numDispDofs>>;
template <typename Scalar>
using DispVector1T = DispVector0T<Scalar>;
template <typename Scalar>
using DispVector2T = DispVector0T<Scalar>;
template <typename Scalar>
using RotVectorT = Dune::BlockVector<Dune::FieldVector<Scalar, numRotDofs>>;
template <typename Scalar>
using SPresVectorT = Dune::BlockVector<Dune::FieldVector<Scalar, numSolidPresDofs>>;

//! \brief The vector the Krylov solver and the preconditioner operate on.
template <typename Scalar>
using TpsaMultiVector = Dune::MultiTypeBlockVector<DispVector0T<Scalar>,
                                                   DispVector1T<Scalar>,
                                                   DispVector2T<Scalar>,
                                                   RotVectorT<Scalar>,
                                                   SPresVectorT<Scalar>>;

// ---------------------------------------------------------------------------
// TpsaMatrixView: a lightweight, non-owning 5x5 view over the sub-matrices
// owned by TpsaMatrix.  Provides the operator interface required by
// Dune::MatrixAdapter / Dune::OverlappingSchwarzOperator (mv, umv, usmv, N, M,
// field_type) plus sub-block access via the S[_i][_j] index syntax used by
// TpsaPreconditioner.
// ---------------------------------------------------------------------------
template <typename Scalar> struct TpsaMatrixRow0;
template <typename Scalar> struct TpsaMatrixRow1;
template <typename Scalar> struct TpsaMatrixRow2;
template <typename Scalar> struct TpsaMatrixRow3;
template <typename Scalar> struct TpsaMatrixRow4;

template <typename Scalar>
class TpsaMatrixView
{
public:
    using size_type = std::size_t;
    using field_type = Scalar;
    using block_type = typename RotRotMatrixT<Scalar>::block_type;

    static constexpr size_type N()
    { return numTpsaFields; }

    static constexpr size_type M()
    { return numTpsaFields; }

    // Row 0: u_x
    const DispDispMatrix00T<Scalar>* M11_00 = nullptr;
    const DispRotMatrix0T<Scalar>* M12_00 = nullptr;
    const DispSPresMatrix0T<Scalar>* M13_00 = nullptr;

    // Row 1: u_y
    const DispDispMatrix11T<Scalar>* M11_11 = nullptr;
    const DispRotMatrix1T<Scalar>* M12_10 = nullptr;
    const DispSPresMatrix1T<Scalar>* M13_10 = nullptr;

    // Row 2: u_z
    const DispDispMatrix22T<Scalar>* M11_22 = nullptr;
    const DispRotMatrix2T<Scalar>* M12_20 = nullptr;
    const DispSPresMatrix2T<Scalar>* M13_20 = nullptr;

    // Row 3: rotation
    const RotDispMatrix0T<Scalar>* M21_00 = nullptr;
    const RotDispMatrix1T<Scalar>* M21_01 = nullptr;
    const RotDispMatrix2T<Scalar>* M21_02 = nullptr;
    const RotRotMatrixT<Scalar>* M22 = nullptr;
    const RotSPresMatrixT<Scalar>* M23 = nullptr;

    // Row 4: solid pressure
    const SPresDispMatrix0T<Scalar>* M31_00 = nullptr;
    const SPresDispMatrix1T<Scalar>* M31_01 = nullptr;
    const SPresDispMatrix2T<Scalar>* M31_02 = nullptr;
    const SPresRotMatrixT<Scalar>* M32 = nullptr;
    const SPresSPresMatrixT<Scalar>* M33 = nullptr;

    //! \brief Sub-block access, S[_i][_j].
    TpsaMatrixRow0<Scalar> operator[](Dune::index_constant<0>) const;
    TpsaMatrixRow1<Scalar> operator[](Dune::index_constant<1>) const;
    TpsaMatrixRow2<Scalar> operator[](Dune::index_constant<2>) const;
    TpsaMatrixRow3<Scalar> operator[](Dune::index_constant<3>) const;
    TpsaMatrixRow4<Scalar> operator[](Dune::index_constant<4>) const;

    //! \brief y = S*x
    void mv(const TpsaMultiVector<Scalar>& x, TpsaMultiVector<Scalar>& y) const
    {
        using namespace Dune::Indices;
        M11_00->mv(x[_0], y[_0]);
        M12_00->umv(x[_3], y[_0]);
        M13_00->umv(x[_4], y[_0]);

        M11_11->mv(x[_1], y[_1]);
        M12_10->umv(x[_3], y[_1]);
        M13_10->umv(x[_4], y[_1]);

        M11_22->mv(x[_2], y[_2]);
        M12_20->umv(x[_3], y[_2]);
        M13_20->umv(x[_4], y[_2]);

        M21_00->mv(x[_0], y[_3]);
        M21_01->umv(x[_1], y[_3]);
        M21_02->umv(x[_2], y[_3]);
        M22->umv(x[_3], y[_3]);
        M23->umv(x[_4], y[_3]);

        M31_00->mv(x[_0], y[_4]);
        M31_01->umv(x[_1], y[_4]);
        M31_02->umv(x[_2], y[_4]);
        M32->umv(x[_3], y[_4]);
        M33->umv(x[_4], y[_4]);
    }

    //! \brief y += S*x
    void umv(const TpsaMultiVector<Scalar>& x, TpsaMultiVector<Scalar>& y) const
    {
        using namespace Dune::Indices;
        M11_00->umv(x[_0], y[_0]);
        M12_00->umv(x[_3], y[_0]);
        M13_00->umv(x[_4], y[_0]);

        M11_11->umv(x[_1], y[_1]);
        M12_10->umv(x[_3], y[_1]);
        M13_10->umv(x[_4], y[_1]);

        M11_22->umv(x[_2], y[_2]);
        M12_20->umv(x[_3], y[_2]);
        M13_20->umv(x[_4], y[_2]);

        M21_00->umv(x[_0], y[_3]);
        M21_01->umv(x[_1], y[_3]);
        M21_02->umv(x[_2], y[_3]);
        M22->umv(x[_3], y[_3]);
        M23->umv(x[_4], y[_3]);

        M31_00->umv(x[_0], y[_4]);
        M31_01->umv(x[_1], y[_4]);
        M31_02->umv(x[_2], y[_4]);
        M32->umv(x[_3], y[_4]);
        M33->umv(x[_4], y[_4]);
    }

    //! \brief y += alpha*S*x
    void usmv(field_type alpha, const TpsaMultiVector<Scalar>& x, TpsaMultiVector<Scalar>& y) const
    {
        using namespace Dune::Indices;
        M11_00->usmv(alpha, x[_0], y[_0]);
        M12_00->usmv(alpha, x[_3], y[_0]);
        M13_00->usmv(alpha, x[_4], y[_0]);

        M11_11->usmv(alpha, x[_1], y[_1]);
        M12_10->usmv(alpha, x[_3], y[_1]);
        M13_10->usmv(alpha, x[_4], y[_1]);

        M11_22->usmv(alpha, x[_2], y[_2]);
        M12_20->usmv(alpha, x[_3], y[_2]);
        M13_20->usmv(alpha, x[_4], y[_2]);

        M21_00->usmv(alpha, x[_0], y[_3]);
        M21_01->usmv(alpha, x[_1], y[_3]);
        M21_02->usmv(alpha, x[_2], y[_3]);
        M22->usmv(alpha, x[_3], y[_3]);
        M23->usmv(alpha, x[_4], y[_3]);

        M31_00->usmv(alpha, x[_0], y[_4]);
        M31_01->usmv(alpha, x[_1], y[_4]);
        M31_02->usmv(alpha, x[_2], y[_4]);
        M32->usmv(alpha, x[_3], y[_4]);
        M33->usmv(alpha, x[_4], y[_4]);
    }
};

// Row proxies for S[row][col].  Only the stored couplings have an overload.
template <typename Scalar>
struct TpsaMatrixRow0
{
    const DispDispMatrix00T<Scalar>& M11_00;
    const DispRotMatrix0T<Scalar>& M12_00;
    const DispSPresMatrix0T<Scalar>& M13_00;

    const DispDispMatrix00T<Scalar>& operator[](Dune::index_constant<0>) const { return M11_00; }
    const DispRotMatrix0T<Scalar>& operator[](Dune::index_constant<3>) const { return M12_00; }
    const DispSPresMatrix0T<Scalar>& operator[](Dune::index_constant<4>) const { return M13_00; }
};

template <typename Scalar>
struct TpsaMatrixRow1
{
    const DispDispMatrix11T<Scalar>& M11_11;
    const DispRotMatrix1T<Scalar>& M12_10;
    const DispSPresMatrix1T<Scalar>& M13_10;

    const DispDispMatrix11T<Scalar>& operator[](Dune::index_constant<1>) const { return M11_11; }
    const DispRotMatrix1T<Scalar>& operator[](Dune::index_constant<3>) const { return M12_10; }
    const DispSPresMatrix1T<Scalar>& operator[](Dune::index_constant<4>) const { return M13_10; }
};

template <typename Scalar>
struct TpsaMatrixRow2
{
    const DispDispMatrix22T<Scalar>& M11_22;
    const DispRotMatrix2T<Scalar>& M12_20;
    const DispSPresMatrix2T<Scalar>& M13_20;

    const DispDispMatrix22T<Scalar>& operator[](Dune::index_constant<2>) const { return M11_22; }
    const DispRotMatrix2T<Scalar>& operator[](Dune::index_constant<3>) const { return M12_20; }
    const DispSPresMatrix2T<Scalar>& operator[](Dune::index_constant<4>) const { return M13_20; }
};

template <typename Scalar>
struct TpsaMatrixRow3
{
    const RotDispMatrix0T<Scalar>& M21_00;
    const RotDispMatrix1T<Scalar>& M21_01;
    const RotDispMatrix2T<Scalar>& M21_02;
    const RotRotMatrixT<Scalar>& M22;
    const RotSPresMatrixT<Scalar>& M23;

    const RotDispMatrix0T<Scalar>& operator[](Dune::index_constant<0>) const { return M21_00; }
    const RotDispMatrix1T<Scalar>& operator[](Dune::index_constant<1>) const { return M21_01; }
    const RotDispMatrix2T<Scalar>& operator[](Dune::index_constant<2>) const { return M21_02; }
    const RotRotMatrixT<Scalar>& operator[](Dune::index_constant<3>) const { return M22; }
    const RotSPresMatrixT<Scalar>& operator[](Dune::index_constant<4>) const { return M23; }
};

template <typename Scalar>
struct TpsaMatrixRow4
{
    const SPresDispMatrix0T<Scalar>& M31_00;
    const SPresDispMatrix1T<Scalar>& M31_01;
    const SPresDispMatrix2T<Scalar>& M31_02;
    const SPresRotMatrixT<Scalar>& M32;
    const SPresSPresMatrixT<Scalar>& M33;

    const SPresDispMatrix0T<Scalar>& operator[](Dune::index_constant<0>) const { return M31_00; }
    const SPresDispMatrix1T<Scalar>& operator[](Dune::index_constant<1>) const { return M31_01; }
    const SPresDispMatrix2T<Scalar>& operator[](Dune::index_constant<2>) const { return M31_02; }
    const SPresRotMatrixT<Scalar>& operator[](Dune::index_constant<3>) const { return M32; }
    const SPresSPresMatrixT<Scalar>& operator[](Dune::index_constant<4>) const { return M33; }
};

template <typename Scalar>
TpsaMatrixRow0<Scalar> TpsaMatrixView<Scalar>::operator[](Dune::index_constant<0>) const
{ return {*M11_00, *M12_00, *M13_00}; }

template <typename Scalar>
TpsaMatrixRow1<Scalar> TpsaMatrixView<Scalar>::operator[](Dune::index_constant<1>) const
{ return {*M11_11, *M12_10, *M13_10}; }

template <typename Scalar>
TpsaMatrixRow2<Scalar> TpsaMatrixView<Scalar>::operator[](Dune::index_constant<2>) const
{ return {*M11_22, *M12_20, *M13_20}; }

template <typename Scalar>
TpsaMatrixRow3<Scalar> TpsaMatrixView<Scalar>::operator[](Dune::index_constant<3>) const
{ return {*M21_00, *M21_01, *M21_02, *M22, *M23}; }

template <typename Scalar>
TpsaMatrixRow4<Scalar> TpsaMatrixView<Scalar>::operator[](Dune::index_constant<4>) const
{ return {*M31_00, *M31_01, *M31_02, *M32, *M33}; }

} // namespace Opm::Linear

namespace Dune {

// Dune::MultiTypeBlockVector needs the field traits of its components to form
// scalar products and norms.
template<>
struct FieldTraits<BlockVector<FieldVector<double, 3>>>
{
    using field_type = double;
    using real_type = double;
};

template<>
struct FieldTraits<BlockVector<FieldVector<float, 3>>>
{
    using field_type = float;
    using real_type = float;
};

template<>
struct FieldTraits<BlockVector<FieldVector<double, 1>>>
{
    using field_type = double;
    using real_type = double;
};

template<>
struct FieldTraits<BlockVector<FieldVector<float, 1>>>
{
    using field_type = float;
    using real_type = float;
};

} // namespace Dune

#endif // OPM_TPSA_TYPES_HPP
