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
#ifndef OPM_TPSA_VECTOR_HPP
#define OPM_TPSA_VECTOR_HPP

#include <opm/simulators/linalg/tpsa/TpsaTypes.hpp>

#include <dune/common/fvector.hh>
#include <dune/common/hybridutilities.hh>
#include <dune/common/indices.hh>

#include <cmath>
#include <cstddef>

namespace Opm::Linear
{

/*!
 * \brief Field-split residual/update vector for the TPSA system.
 *
 * Storage is a Dune::MultiTypeBlockVector of the five TPSA fields (see
 * TpsaTypes.hpp), which is what the Krylov solver and the block preconditioner
 * operate on.  To the linearizer and the Newton method the class still looks
 * like a Dune::BlockVector of 7-component blocks: `v[i]` yields the seven
 * equations of cell `i`, either as a value (const access) or as a proxy that
 * scatters into the five sub-vectors (mutable access).
 */
template <class ScalarT>
class TpsaVector
{
public:
    using Scalar = ScalarT;
    using field_type = ScalarT;
    using EqVector = Dune::FieldVector<Scalar, numTpsaEq>;
    using block_type = EqVector;
    using IstlVector = TpsaMultiVector<Scalar>;
    using size_type = std::size_t;

    /*!
     * \brief Mutable reference to the seven equations of a single cell.
     *
     * Reads and writes are scattered over the five sub-vectors.
     */
    class EntryProxy
    {
    public:
        EntryProxy(IstlVector& v, std::size_t dofIdx)
            : v_(&v)
              , i_(dofIdx)
        {
        }

        static constexpr std::size_t size()
        {
            return numTpsaEq;
        }

        Scalar& operator[](std::size_t eqIdx)
        {
            return at_(eqIdx);
        }

        Scalar operator[](std::size_t eqIdx) const
        {
            return at_(eqIdx);
        }

        /*!
         * \brief Gathers to an EqVector block vector
         *
         * \warning This function requires an explicit return type EqVector, contrary to,
         * "auto b = v[i]" which returns a EntryProxy type
         */
        operator EqVector() const
        {
            EqVector res;
            for (std::size_t eqIdx = 0; eqIdx < numTpsaEq; ++eqIdx) {
                res[eqIdx] = at_(eqIdx);
            }

            return res;
        }

        EntryProxy& operator=(Scalar value)
        {
            for (std::size_t eqIdx = 0; eqIdx < numTpsaEq; ++eqIdx) {
                at_(eqIdx) = value;
            }

            return *this;
        }

        EntryProxy& operator=(const EqVector& value)
        {
            for (std::size_t eqIdx = 0; eqIdx < numTpsaEq; ++eqIdx) {
                at_(eqIdx) = value[eqIdx];
            }

            return *this;
        }

        EntryProxy& operator+=(const EqVector& value)
        {
            for (std::size_t eqIdx = 0; eqIdx < numTpsaEq; ++eqIdx) {
                at_(eqIdx) += value[eqIdx];
            }

            return *this;
        }

        EntryProxy& operator-=(const EqVector& value)
        {
            for (std::size_t eqIdx = 0; eqIdx < numTpsaEq; ++eqIdx) {
                at_(eqIdx) -= value[eqIdx];
            }

            return *this;
        }

        EntryProxy& operator*=(Scalar factor)
        {
            for (std::size_t eqIdx = 0; eqIdx < numTpsaEq; ++eqIdx) {
                at_(eqIdx) *= factor;
            }

            return *this;
        }

    private:
        Scalar& at_(std::size_t eqIdx) const
        {
            using namespace Dune::Indices;
            switch (eqIdx) {
            case 0:
                return (*v_)[_0][i_][0];
            case 1:
                return (*v_)[_1][i_][0];
            case 2:
                return (*v_)[_2][i_][0];
            case 3:
                return (*v_)[_3][i_][0];
            case 4:
                return (*v_)[_3][i_][1];
            case 5:
                return (*v_)[_3][i_][2];
            default:
                return (*v_)[_4][i_][0];
            }
        }

        IstlVector* v_;
        std::size_t i_;
    };

    TpsaVector() = default;

    explicit TpsaVector(std::size_t numDof)
    {
        resize(numDof);
    }

    void resize(std::size_t numDof)
    {
        using namespace Dune::Indices;
        Dune::Hybrid::forEach(Dune::range(Dune::index_constant<numTpsaFields>{}),
                              [&](auto fieldIdx) {
                                  v_[fieldIdx].resize(numDof);
                              });
        size_ = numDof;
    }

    std::size_t size() const
    {
        return size_;
    }

    std::size_t N() const
    {
        return size_;
    }

    TpsaVector& operator=(Scalar value)
    {
        Dune::Hybrid::forEach(Dune::range(Dune::index_constant<numTpsaFields>{}),
                              [&](auto fieldIdx) {
                                  v_[fieldIdx] = value;
                              });

        return *this;
    }

    TpsaVector& operator+=(const TpsaVector& other)
    {
        Dune::Hybrid::forEach(Dune::range(Dune::index_constant<numTpsaFields>{}),
                              [&](auto fieldIdx) {
                                  v_[fieldIdx] += other.v_[fieldIdx];
                              });

        return *this;
    }

    TpsaVector& operator-=(const TpsaVector& other)
    {
        Dune::Hybrid::forEach(Dune::range(Dune::index_constant<numTpsaFields>{}),
                              [&](auto fieldIdx) {
                                  v_[fieldIdx] -= other.v_[fieldIdx];
                              });

        return *this;
    }

    TpsaVector& operator*=(Scalar factor)
    {
        Dune::Hybrid::forEach(Dune::range(Dune::index_constant<numTpsaFields>{}),
                              [&](auto fieldIdx) {
                                  v_[fieldIdx] *= factor;
                              });

        return *this;
    }

    Scalar one_norm() const
    {
        Scalar norm = 0.0;
        Dune::Hybrid::forEach(Dune::range(Dune::index_constant<numTpsaFields>{}),
                              [&](auto fieldIdx) {
                                  norm += v_[fieldIdx].one_norm();
                              });

        return norm;
    }

    Scalar two_norm2() const
    {
        Scalar norm = 0.0;
        Dune::Hybrid::forEach(Dune::range(Dune::index_constant<numTpsaFields>{}),
                              [&](auto fieldIdx) {
                                  norm += v_[fieldIdx].two_norm2();
                              });

        return norm;
    }

    Scalar two_norm() const
    {
        return std::sqrt(two_norm2());
    }

    Scalar infinity_norm() const
    {
        Scalar norm = 0.0;
        Dune::Hybrid::forEach(Dune::range(Dune::index_constant<numTpsaFields>{}),
                              [&](auto fieldIdx) {
                                  norm = std::max(norm, v_[fieldIdx].infinity_norm());
                              });

        return norm;
    }

    //! \brief The seven equations of cell dofIdx, gathered into a dense block.
    EqVector operator[](std::size_t dofIdx) const
    {
        using namespace Dune::Indices;
        EqVector res;
        res[0] = v_[_0][dofIdx][0];
        res[1] = v_[_1][dofIdx][0];
        res[2] = v_[_2][dofIdx][0];
        res[3] = v_[_3][dofIdx][0];
        res[4] = v_[_3][dofIdx][1];
        res[5] = v_[_3][dofIdx][2];
        res[6] = v_[_4][dofIdx][0];

        return res;
    }

    //! \brief Writable handle on the seven equations of cell dofIdx.
    EntryProxy operator[](std::size_t dofIdx)
    {
        return EntryProxy(v_, dofIdx);
    }

    //! \brief The underlying multi-type vector handed to the linear solver.
    IstlVector& istlVector()
    {
        return v_;
    }

    const IstlVector& istlVector() const
    {
        return v_;
    }

private:
    IstlVector v_{};
    std::size_t size_{0};
};

} // namespace Opm::Linear

#endif // OPM_TPSA_VECTOR_HPP
