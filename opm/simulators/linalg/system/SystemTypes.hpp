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
template<typename Scalar>
using DispDispMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numDispDofs> >;
template<typename Scalar>
using RotRotMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numRotDofs, numRotDofs> >;
template <typename Scalar>
using SPresSPresMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numSolidPresDofs, numSolidPresDofs> >;

// Off-diagonal matrix types
template <typename Scalar>
using DispRotMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numRotDofs> >;
template <typename Scalar>
using DispSPresMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numDispDofs, numSolidPresDofs> >;

template<typename Scalar>
using RotDispMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numRotDofs, numDispDofs> >;
template<typename Scalar>
using RotSPresMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numRotDofs, numSolidPresDofs> >;

template<typename Scalar>
using SPresDispMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numSolidPresDofs, numDispDofs> >;
template<typename Scalar>
using SPresRotMatrixT = Linear::IstlSparseMatrixAdapter<
    MatrixBlock<Scalar, numSolidPresDofs, numRotDofs> >;

// Vector types
template<typename Scalar>
using DispVectorT = Dune::BlockVector<Dune::FieldVector<Scalar, numDispDofs> >;
template<typename Scalar>
using RotVectorT = Dune::BlockVector<Dune::FieldVector<Scalar, numRotDofs>>;
template<typename Scalar>
using SPresVectorT = Dune::BlockVector<Dune::FieldVector<Scalar, numSolidPresDofs>>;
template<typename Scalar>
using SystemVectorT = Dune::MultiTypeBlockVector<DispVectorT<Scalar>,
                                                 RotVectorT<Scalar>,
                                                 SPresVectorT<Scalar> >;

template <typename Scalar>
struct SystemMatrixRow0T;
template <typename Scalar>
struct SystemMatrixRow1T;
template <typename Scalar>
struct SystemMatrixRow2T;

template<typename Scalar>
class SystemMatrixT
{
public:
    using size_type  = std::size_t;
    using field_type = Scalar;
    using block_type = DispDispMatrixT<Scalar>::IstlMatrix::block_type; // Generalize???

    static constexpr size_type N() { return 3; }
    static constexpr size_type M() { return 3; }

    const DispDispMatrixT<Scalar>* M11 = nullptr;
    const DispRotMatrixT<Scalar>* M12 = nullptr;
    const DispSPresMatrixT<Scalar>* M13 = nullptr;

    const RotDispMatrixT<Scalar>* M21 = nullptr;
    const RotRotMatrixT<Scalar>* M22 = nullptr;
    const RotSPresMatrixT<Scalar>* M23 = nullptr;

    const SPresDispMatrixT<Scalar>* M31 = nullptr;
    const SPresRotMatrixT<Scalar>* M32 = nullptr;
    const SPresSPresMatrixT<Scalar>* M33 = nullptr;

    // Sub-block access
    SystemMatrixRow0T<Scalar> operator[](Dune::index_constant<0>) const;

    SystemMatrixRow1T<Scalar> operator[](Dune::index_constant<1>) const;

    SystemMatrixRow2T<Scalar> operator[](Dune::index_constant<2>) const;

    // Matrix-vector products required by Dune linear operators.
    void mv(const SystemVectorT<Scalar>& x, SystemVectorT<Scalar>& y) const
    {
        using namespace Dune::Indices;
        M11->istlMatrix().mv(x[_0], y[_0]);
        M12->istlMatrix().umv(x[_1], y[_0]);
        M13->istlMatrix().umv(x[_2], y[_0]);

        M21->istlMatrix().mv(x[_0], y[_1]);
        M22->istlMatrix().umv(x[_1], y[_1]);
        M23->istlMatrix().umv(x[_2], y[_1]);

        M31->istlMatrix().mv(x[_0], y[_2]);
        M32->istlMatrix().umv(x[_1], y[_2]);
        M33->istlMatrix().umv(x[_2], y[_2]);
    }

    void umv(const SystemVectorT<Scalar>& x, SystemVectorT<Scalar>& y) const
    {
        using namespace Dune::Indices;
        M11->istlMatrix().umv(x[_0], y[_0]);
        M12->istlMatrix().umv(x[_1], y[_0]);
        M13->istlMatrix().umv(x[_2], y[_0]);

        M21->istlMatrix().umv(x[_0], y[_1]);
        M22->istlMatrix().umv(x[_1], y[_1]);
        M23->istlMatrix().umv(x[_2], y[_1]);

        M31->istlMatrix().umv(x[_0], y[_2]);
        M32->istlMatrix().umv(x[_1], y[_2]);
        M33->istlMatrix().umv(x[_2], y[_2]);
    }

    void usmv(field_type alpha, const SystemVectorT<Scalar>& x, SystemVectorT<Scalar>& y) const
    {
        using namespace Dune::Indices;
        M11->istlMatrix().usmv(alpha, x[_0], y[_0]);
        M12->istlMatrix().usmv(alpha, x[_1], y[_0]);
        M13->istlMatrix().usmv(alpha, x[_2], y[_0]);

        M21->istlMatrix().usmv(alpha, x[_0], y[_1]);
        M22->istlMatrix().usmv(alpha, x[_1], y[_1]);
        M23->istlMatrix().usmv(alpha, x[_2], y[_1]);

        M31->istlMatrix().usmv(alpha, x[_0], y[_2]);
        M32->istlMatrix().usmv(alpha, x[_1], y[_2]);
        M33->istlMatrix().usmv(alpha, x[_2], y[_2]);
    }
};

// Row proxies for  S[row][col]  — simple aggregates, no back-pointers.
template<typename Scalar>
struct SystemMatrixRow0T
{
    const DispDispMatrixT<Scalar>& M11;
    const DispRotMatrixT<Scalar>& M12;
    const DispSPresMatrixT<Scalar>& M13;
    const DispDispMatrixT<Scalar>& operator[](Dune::index_constant<0>) const { return M11; }
    const DispRotMatrixT<Scalar>& operator[](Dune::index_constant<1>) const { return M12; }
    const DispSPresMatrixT<Scalar>& operator[](Dune::index_constant<2>) const { return M13; }
};

template<typename Scalar>
struct SystemMatrixRow1T
{
    const RotDispMatrixT<Scalar>& M21;
    const RotRotMatrixT<Scalar>& M22;
    const RotSPresMatrixT<Scalar>& M23;
    const RotDispMatrixT<Scalar>& operator[](Dune::index_constant<0>) const { return M21; }
    const RotRotMatrixT<Scalar>& operator[](Dune::index_constant<1>) const { return M22; }
    const RotSPresMatrixT<Scalar>& operator[](Dune::index_constant<2>) const { return M23; }
};

template<typename Scalar>
struct SystemMatrixRow2T
{
    const SPresDispMatrixT<Scalar>& M31;
    const SPresRotMatrixT<Scalar>& M32;
    const SPresSPresMatrixT<Scalar>& M33;
    const SPresDispMatrixT<Scalar>& operator[](Dune::index_constant<0>) const { return M31; }
    const SPresRotMatrixT<Scalar>& operator[](Dune::index_constant<1>) const { return M32; }
    const SPresSPresMatrixT<Scalar>& operator[](Dune::index_constant<2>) const { return M33; }
};

template<typename Scalar>
SystemMatrixRow0T<Scalar> SystemMatrixT<Scalar>::operator[](Dune::index_constant<0>) const
{ return {*M11, *M12, *M13}; }

template<typename Scalar>
SystemMatrixRow1T<Scalar> SystemMatrixT<Scalar>::operator[](Dune::index_constant<1>) const
{ return {*M21, *M22, *M23}; }

template<typename Scalar>
SystemMatrixRow2T<Scalar> SystemMatrixT<Scalar>::operator[](Dune::index_constant<2>) const
{ return {*M31, *M32, *M33}; }

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
