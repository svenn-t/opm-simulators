#pragma once

#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/multitypeblockmatrix.hh>
#include <dune/istl/multitypeblockvector.hh>

#include <opm/simulators/linalg/matrixblock.hh>
#include <opm/simulators/linalg/istlsparsematrixadapter.hh>

namespace Opm
{
inline constexpr int numDispDofs = 1;
inline constexpr int numRotDofs = 3;
inline constexpr int numSolidPresDofs = 1;

// Diagonal matrix types
template <typename Scalar>
using DispDispMatrix00T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numDispDofs> >;
template <typename Scalar>
using DispDispMatrix11T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numDispDofs> >;
template <typename Scalar>
using DispDispMatrix22T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numDispDofs> >;

template <typename Scalar>
using RotRotMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numRotDofs, numRotDofs> >;

template <typename Scalar>
using SPresSPresMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numSolidPresDofs, numSolidPresDofs> >;

// Off-diagonal matrix types
template <typename Scalar>
using DispRotMatrix0T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numRotDofs> >;
template <typename Scalar>
using DispRotMatrix1T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numRotDofs> >;
template <typename Scalar>
using DispRotMatrix2T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numRotDofs> >;
template <typename Scalar>
using DispSPresMatrix0T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numSolidPresDofs> >;
template <typename Scalar>
using DispSPresMatrix1T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numSolidPresDofs> >;
template <typename Scalar>
using DispSPresMatrix2T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numSolidPresDofs> >;


template <typename Scalar>
using RotDispMatrix0T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numRotDofs, numDispDofs> >;
template <typename Scalar>
using RotDispMatrix1T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numRotDofs, numDispDofs> >;
template <typename Scalar>
using RotDispMatrix2T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numRotDofs, numDispDofs> >;
template <typename Scalar>
using RotSPresMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numRotDofs, numSolidPresDofs> >;

template <typename Scalar>
using SPresDispMatrix0T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numSolidPresDofs, numDispDofs> >;
template <typename Scalar>
using SPresDispMatrix1T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numSolidPresDofs, numDispDofs> >;
template <typename Scalar>
using SPresDispMatrix2T = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numSolidPresDofs, numDispDofs> >;
template <typename Scalar>
using SPresRotMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numSolidPresDofs, numRotDofs> >;

// Vector types
template <typename Scalar>
using DispVector0T = Dune::BlockVector<Dune::FieldVector<Scalar, numDispDofs> >;
template <typename Scalar>
using DispVector1T = Dune::BlockVector<Dune::FieldVector<Scalar, numDispDofs> >;
template <typename Scalar>
using DispVector2T = Dune::BlockVector<Dune::FieldVector<Scalar, numDispDofs> >;
template <typename Scalar>
using RotVectorT = Dune::BlockVector<Dune::FieldVector<Scalar, numRotDofs> >;
template <typename Scalar>
using SPresVectorT = Dune::BlockVector<Dune::FieldVector<Scalar, numSolidPresDofs> >;

template <typename Scalar>
using SystemVectorT = Dune::MultiTypeBlockVector<
    DispVector0T<Scalar>,
    DispVector1T<Scalar>,
    DispVector2T<Scalar>,
    RotVectorT<Scalar>,
    SPresVectorT<Scalar> >;

template <typename Scalar>
struct SystemMatrixRow0T;
template <typename Scalar>
struct SystemMatrixRow1T;
template <typename Scalar>
struct SystemMatrixRow2T;
template <typename Scalar>
struct SystemMatrixRow3T;
template <typename Scalar>
struct SystemMatrixRow4T;

template <typename Scalar>
class SystemMatrixT
{
public:
    using size_type = std::size_t;
    using field_type = Scalar;
    using block_type = RotRotMatrixT<Scalar>::IstlMatrix::block_type; // Generalize???

    static constexpr size_type N()
    {
        return 5;
    }

    static constexpr size_type M()
    {
        return 5;
    }

    const DispDispMatrix00T<Scalar>* M11_00 = nullptr;
    const DispDispMatrix11T<Scalar>* M11_11 = nullptr;
    const DispDispMatrix22T<Scalar>* M11_22 = nullptr;

    const DispRotMatrix0T<Scalar>* M12_00 = nullptr;
    const DispRotMatrix1T<Scalar>* M12_10 = nullptr;
    const DispRotMatrix2T<Scalar>* M12_20 = nullptr;

    const DispSPresMatrix0T<Scalar>* M13_00 = nullptr;
    const DispSPresMatrix1T<Scalar>* M13_10 = nullptr;
    const DispSPresMatrix2T<Scalar>* M13_20 = nullptr;

    const RotDispMatrix0T<Scalar>* M21_00 = nullptr;
    const RotDispMatrix1T<Scalar>* M21_01 = nullptr;
    const RotDispMatrix2T<Scalar>* M21_02 = nullptr;

    const RotRotMatrixT<Scalar>* M22 = nullptr;

    const RotSPresMatrixT<Scalar>* M23 = nullptr;

    const SPresDispMatrix0T<Scalar>* M31_00 = nullptr;
    const SPresDispMatrix1T<Scalar>* M31_01 = nullptr;
    const SPresDispMatrix2T<Scalar>* M31_02 = nullptr;

    const SPresRotMatrixT<Scalar>* M32 = nullptr;

    const SPresSPresMatrixT<Scalar>* M33 = nullptr;

    // Sub-block access
    SystemMatrixRow0T<Scalar> operator[](Dune::index_constant<0>) const;

    SystemMatrixRow1T<Scalar> operator[](Dune::index_constant<1>) const;

    SystemMatrixRow2T<Scalar> operator[](Dune::index_constant<2>) const;

    SystemMatrixRow3T<Scalar> operator[](Dune::index_constant<3>) const;

    SystemMatrixRow4T<Scalar> operator[](Dune::index_constant<4>) const;

    // Matrix-vector products required by Dune linear operators.
    void mv(const SystemVectorT<Scalar>& x, SystemVectorT<Scalar>& y) const
    {
        using namespace Dune::Indices;
        M11_00->istlMatrix().mv(x[_0], y[_0]);
        M12_00->istlMatrix().umv(x[_3], y[_0]);
        M13_00->istlMatrix().umv(x[_4], y[_0]);

        M11_11->istlMatrix().mv(x[_1], y[_1]);
        M12_10->istlMatrix().umv(x[_3], y[_1]);
        M13_10->istlMatrix().umv(x[_4], y[_1]);

        M11_22->istlMatrix().mv(x[_2], y[_2]);
        M12_20->istlMatrix().umv(x[_3], y[_2]);
        M13_20->istlMatrix().umv(x[_4], y[_2]);

        M21_00->istlMatrix().mv(x[_0], y[_3]);
        M21_01->istlMatrix().umv(x[_1], y[_3]);
        M21_02->istlMatrix().umv(x[_2], y[_3]);
        M22->istlMatrix().umv(x[_3], y[_3]);
        M23->istlMatrix().umv(x[_4], y[_3]);

        M31_00->istlMatrix().mv(x[_0], y[_4]);
        M31_01->istlMatrix().umv(x[_1], y[_4]);
        M31_02->istlMatrix().umv(x[_2], y[_4]);
        M32->istlMatrix().umv(x[_3], y[_4]);
        M33->istlMatrix().umv(x[_4], y[_4]);
    }

    void umv(const SystemVectorT<Scalar>& x, SystemVectorT<Scalar>& y) const
    {
        using namespace Dune::Indices;
        M11_00->istlMatrix().umv(x[_0], y[_0]);
        M12_00->istlMatrix().umv(x[_3], y[_0]);
        M13_00->istlMatrix().umv(x[_4], y[_0]);

        M11_11->istlMatrix().umv(x[_1], y[_1]);
        M12_10->istlMatrix().umv(x[_3], y[_1]);
        M13_10->istlMatrix().umv(x[_4], y[_1]);

        M11_22->istlMatrix().umv(x[_2], y[_2]);
        M12_20->istlMatrix().umv(x[_3], y[_2]);
        M13_20->istlMatrix().umv(x[_4], y[_2]);

        M21_00->istlMatrix().umv(x[_0], y[_3]);
        M21_01->istlMatrix().umv(x[_1], y[_3]);
        M21_02->istlMatrix().umv(x[_2], y[_3]);
        M22->istlMatrix().umv(x[_3], y[_3]);
        M23->istlMatrix().umv(x[_4], y[_3]);

        M31_00->istlMatrix().umv(x[_0], y[_4]);
        M31_01->istlMatrix().umv(x[_1], y[_4]);
        M31_02->istlMatrix().umv(x[_2], y[_4]);
        M32->istlMatrix().umv(x[_3], y[_4]);
        M33->istlMatrix().umv(x[_4], y[_4]);
    }

    void usmv(field_type alpha, const SystemVectorT<Scalar>& x, SystemVectorT<Scalar>& y) const
    {
        using namespace Dune::Indices;
        M11_00->istlMatrix().usmv(alpha, x[_0], y[_0]);
        M12_00->istlMatrix().usmv(alpha, x[_3], y[_0]);
        M13_00->istlMatrix().usmv(alpha, x[_4], y[_0]);

        M11_11->istlMatrix().usmv(alpha, x[_1], y[_1]);
        M12_10->istlMatrix().usmv(alpha, x[_3], y[_1]);
        M13_10->istlMatrix().usmv(alpha, x[_4], y[_1]);

        M11_22->istlMatrix().usmv(alpha, x[_2], y[_2]);
        M12_20->istlMatrix().usmv(alpha, x[_3], y[_2]);
        M13_20->istlMatrix().usmv(alpha, x[_4], y[_2]);

        M21_00->istlMatrix().usmv(alpha, x[_0], y[_3]);
        M21_01->istlMatrix().usmv(alpha, x[_1], y[_3]);
        M21_02->istlMatrix().usmv(alpha, x[_2], y[_3]);
        M22->istlMatrix().usmv(alpha, x[_3], y[_3]);
        M23->istlMatrix().usmv(alpha, x[_4], y[_3]);

        M31_00->istlMatrix().usmv(alpha, x[_0], y[_4]);
        M31_01->istlMatrix().usmv(alpha, x[_1], y[_4]);
        M31_02->istlMatrix().usmv(alpha, x[_2], y[_4]);
        M32->istlMatrix().usmv(alpha, x[_3], y[_4]);
        M33->istlMatrix().usmv(alpha, x[_4], y[_4]);
    }
};

template <typename Scalar>
struct SystemMatrixRow0T
{
    const DispDispMatrix00T<Scalar>& M11_00;
    const DispRotMatrix0T<Scalar>& M12_00;
    const DispSPresMatrix0T<Scalar>& M13_00;

    const DispDispMatrix00T<Scalar>& operator[](Dune::index_constant<0>) const
    {
        return M11_00;
    }

    const DispRotMatrix0T<Scalar>& operator[](Dune::index_constant<3>) const
    {
        return M12_00;
    }

    const DispSPresMatrix0T<Scalar>& operator[](Dune::index_constant<4>) const
    {
        return M13_00;
    }
};

template <typename Scalar>
struct SystemMatrixRow1T
{
    const DispDispMatrix11T<Scalar>& M11_11;
    const DispRotMatrix1T<Scalar>& M12_10;
    const DispSPresMatrix1T<Scalar>& M13_10;

    const DispDispMatrix00T<Scalar>& operator[](Dune::index_constant<1>) const
    {
        return M11_11;
    }

    const DispRotMatrix0T<Scalar>& operator[](Dune::index_constant<3>) const
    {
        return M12_10;
    }

    const DispSPresMatrix0T<Scalar>& operator[](Dune::index_constant<4>) const
    {
        return M13_10;
    }
};

template <typename Scalar>
struct SystemMatrixRow2T
{
    const DispDispMatrix22T<Scalar>& M11_22;
    const DispRotMatrix2T<Scalar>& M12_20;
    const DispSPresMatrix2T<Scalar>& M13_20;

    const DispDispMatrix00T<Scalar>& operator[](Dune::index_constant<2>) const
    {
        return M11_22;
    }

    const DispRotMatrix0T<Scalar>& operator[](Dune::index_constant<3>) const
    {
        return M12_20;
    }

    const DispSPresMatrix0T<Scalar>& operator[](Dune::index_constant<4>) const
    {
        return M13_20;
    }
};

template <typename Scalar>
struct SystemMatrixRow3T
{
    const RotDispMatrix0T<Scalar>& M21_00;
    const RotDispMatrix1T<Scalar>& M21_01;
    const RotDispMatrix2T<Scalar>& M21_02;
    const RotRotMatrixT<Scalar>& M22;
    const RotSPresMatrixT<Scalar>& M23;

    const RotDispMatrix0T<Scalar>& operator[](Dune::index_constant<0>) const
    {
        return M21_00;
    }

    const RotDispMatrix0T<Scalar>& operator[](Dune::index_constant<1>) const
    {
        return M21_01;
    }

    const RotDispMatrix0T<Scalar>& operator[](Dune::index_constant<2>) const
    {
        return M21_02;
    }

    const RotRotMatrixT<Scalar>& operator[](Dune::index_constant<3>) const
    {
        return M22;
    }

    const RotSPresMatrixT<Scalar>& operator[](Dune::index_constant<4>) const
    {
        return M23;
    }
};

template <typename Scalar>
struct SystemMatrixRow4T
{
    const SPresDispMatrix0T<Scalar>& M31_00;
    const SPresDispMatrix1T<Scalar>& M31_01;
    const SPresDispMatrix2T<Scalar>& M31_02;
    const SPresRotMatrixT<Scalar>& M32;
    const SPresSPresMatrixT<Scalar>& M33;

    const SPresDispMatrix0T<Scalar>& operator[](Dune::index_constant<0>) const
    {
        return M31_00;
    }

    const SPresDispMatrix1T<Scalar>& operator[](Dune::index_constant<1>) const
    {
        return M31_01;
    }

    const SPresDispMatrix2T<Scalar>& operator[](Dune::index_constant<2>) const
    {
        return M31_02;
    }

    const SPresRotMatrixT<Scalar>& operator[](Dune::index_constant<3>) const
    {
        return M32;
    }

    const SPresSPresMatrixT<Scalar>& operator[](Dune::index_constant<4>) const
    {
        return M33;
    }
};

template <typename Scalar>
SystemMatrixRow0T<Scalar>
SystemMatrixT<Scalar>::operator[](Dune::index_constant<0>) const
{
    return {*M11_00, *M12_00, *M13_00};
}

template <typename Scalar>
SystemMatrixRow1T<Scalar>
SystemMatrixT<Scalar>::operator[](Dune::index_constant<1>) const
{
    return {*M11_11, *M12_10, *M13_10};
}

template <typename Scalar>
SystemMatrixRow2T<Scalar>
SystemMatrixT<Scalar>::operator[](Dune::index_constant<2>) const
{
    return {*M11_22, *M12_20, *M13_20};
}

template <typename Scalar>
SystemMatrixRow3T<Scalar>
SystemMatrixT<Scalar>::operator[](Dune::index_constant<3>) const
{
    return {*M21_00, *M21_01, *M21_02, *M22, *M23};
}

template <typename Scalar>
SystemMatrixRow4T<Scalar>
SystemMatrixT<Scalar>::operator[](Dune::index_constant<4>) const
{
    return {*M31_00, *M31_01, *M31_02, *M32, *M33};
}

} // namespace Opm

namespace Dune
{
// Specialization for field traits of multitypeblockvector
template <>
struct FieldTraits<BlockVector<FieldVector<double, 3> > >
{
    using field_type = double;
    using real_type = double;
};

template <>
struct FieldTraits<BlockVector<FieldVector<float, 3> > >
{
    using field_type = float;
    using real_type = float;
};

template <>
struct FieldTraits<BlockVector<FieldVector<double, 1> > >
{
    using field_type = double;
    using real_type = double;
};

template <>
struct FieldTraits<BlockVector<FieldVector<float, 1> > >
{
    using field_type = float;
    using real_type = float;
};
}
