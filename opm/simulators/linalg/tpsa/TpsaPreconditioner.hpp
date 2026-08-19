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
#ifndef OPM_TPSA_PRECONDITIONER_HPP
#define OPM_TPSA_PRECONDITIONER_HPP

#include <dune/istl/operators.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/paamg/pinfo.hh>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>
#include <opm/simulators/linalg/tpsa/TpsaTypes.hpp>

#include <functional>
#include <memory>
#include <type_traits>

namespace Opm
{

template <typename Scalar>
using SeqDispDispOperatorT = Dune::MatrixAdapter<Linear::DispDispMatrix00T<Scalar>,
                                                 Linear::DispVector0T<Scalar>,
                                                 Linear::DispVector0T<Scalar> >;
template <typename Scalar>
using SeqRotRotOperatorT = Dune::MatrixAdapter<Linear::RotRotMatrixT<Scalar>,
                                               Linear::RotVectorT<Scalar>,
                                               Linear::RotVectorT<Scalar> >;
template <typename Scalar>
using SeqSPresSPresOperatorT = Dune::MatrixAdapter<Linear::SPresSPresMatrixT<Scalar>,
                                                   Linear::SPresVectorT<Scalar>,
                                                   Linear::SPresVectorT<Scalar> >;

#if HAVE_MPI
using TpsaParComm = Dune::OwnerOverlapCopyCommunication<int, int>;

template <typename Scalar>
using ParDispDispOperatorT = Dune::OverlappingSchwarzOperator<Linear::DispDispMatrix00T<Scalar>,
                                                              Linear::DispVector0T<Scalar>,
                                                              Linear::DispVector0T<Scalar>,
                                                              TpsaParComm>;
template <typename Scalar>
using ParRotRotOperatorT = Dune::OverlappingSchwarzOperator<Linear::RotRotMatrixT<Scalar>,
                                                            Linear::RotVectorT<Scalar>,
                                                            Linear::RotVectorT<Scalar>,
                                                            TpsaParComm>;
template <typename Scalar>
using ParSPresSPresOperatorT = Dune::OverlappingSchwarzOperator<Linear::SPresSPresMatrixT<Scalar>,
                                                                Linear::SPresVectorT<Scalar>,
                                                                Linear::SPresVectorT<Scalar>,
                                                                TpsaParComm>;
#endif

/*!
 * \brief Block lower-triangular preconditioner for the field-split TPSA system.
 *
 * Solves the three scalar displacement blocks, carries their contribution over
 * to the rotation and solid pressure defects, and solves those in turn.  Each
 * diagonal block gets its own Dune::FlexibleSolver, configured through the
 * `disp_disp_solver`, `rot_rot_solver` and `spres_spres_solver` sub-trees; the
 * three displacement blocks share the `disp_disp_solver` configuration.
 *
 * NOTE: the parallel path is implemented but has not been validated on more
 * than one rank.
 */
template <class Scalar, class DispOp, class RotOp, class SPresOp,
          class Comm = Dune::Amg::SequentialInformation>
class TpsaPreconditioner
    : public Dune::PreconditionerWithUpdate<Linear::TpsaMultiVector<Scalar>,
                                            Linear::TpsaMultiVector<Scalar> >
{
    using MultiVector = Linear::TpsaMultiVector<Scalar>;
    using DispSolver = Dune::FlexibleSolver<DispOp>;
    using RotSolver = Dune::FlexibleSolver<RotOp>;
    using SPresSolver = Dune::FlexibleSolver<SPresOp>;

public:
    static constexpr bool isParallel = !std::is_same_v<Comm, Dune::Amg::SequentialInformation>;

    static constexpr auto _0 = Dune::Indices::_0;
    static constexpr auto _1 = Dune::Indices::_1;
    static constexpr auto _2 = Dune::Indices::_2;
    static constexpr auto _3 = Dune::Indices::_3;
    static constexpr auto _4 = Dune::Indices::_4;

    //! \brief The sub-solvers have no pressure equation to single out.
    static constexpr std::size_t pressureIdx = 0;

    //! \brief Sequential constructor.
    template <bool P = isParallel, std::enable_if_t<!P, int> = 0>
    TpsaPreconditioner(const Linear::TpsaMatrixView<Scalar>& S, const PropertyTree& prm)
        : S_(S)
    {
        initSubSolvers_(prm);
    }

    //! \brief Parallel constructor.
    template <bool P = isParallel, std::enable_if_t<P, int> = 0>
    TpsaPreconditioner(const Linear::TpsaMatrixView<Scalar>& S,
                       const PropertyTree& prm,
                       const Comm& comm)
        : S_(S)
          , comm_(&comm)
    {
        initSubSolvers_(prm);
    }

    void pre(MultiVector&, MultiVector&) override
    {
    }

    void post(MultiVector&) override
    {
    }

    Dune::SolverCategory::Category category() const override
    {
        if constexpr (isParallel) {
            return Dune::SolverCategory::overlapping;
        } else {
            return Dune::SolverCategory::sequential;
        }
    }

    void update() override
    {
        dispSolver0_->preconditioner().update();
        dispSolver1_->preconditioner().update();
        dispSolver2_->preconditioner().update();
        rotSolver_->preconditioner().update();
        sPresSolver_->preconditioner().update();
    }

    bool hasPerfectUpdate() const override
    {
        return true;
    }

    void apply(MultiVector& v, const MultiVector& d) override
    {
        Dune::InverseOperatorResult result;

        // The defects of the coupled fields are updated as the sweep proceeds,
        // so they need their own copies.
        auto d0 = d[_0];
        auto d1 = d[_1];
        auto d2 = d[_2];
        auto d3 = d[_3];
        auto d4 = d[_4];

        dispSolver0_->apply(v[_0], d0, result);
        dispSolver1_->apply(v[_1], d1, result);
        dispSolver2_->apply(v[_2], d2, result);

        if constexpr (isParallel) {
            // The displacement updates are read on the overlap by the mmv
            // couplings below, so they have to be consistent first.
            comm_->copyOwnerToAll(v[_0], v[_0]);
            comm_->copyOwnerToAll(v[_1], v[_1]);
            comm_->copyOwnerToAll(v[_2], v[_2]);
        }

        S_[_3][_0].mmv(v[_0], d3);
        S_[_3][_1].mmv(v[_1], d3);
        S_[_3][_2].mmv(v[_2], d3);
        rotSolver_->apply(v[_3], d3, result);

        S_[_4][_0].mmv(v[_0], d4);
        S_[_4][_1].mmv(v[_1], d4);
        S_[_4][_2].mmv(v[_2], d4);
        sPresSolver_->apply(v[_4], d4, result);

        if constexpr (isParallel) {
            comm_->copyOwnerToAll(v[_3], v[_3]);
            comm_->copyOwnerToAll(v[_4], v[_4]);
        }
    }

private:
    void initSubSolvers_(const PropertyTree& prm)
    {
        const auto dispPrm = prm.get_child("disp_disp_solver");
        const auto rotPrm = prm.get_child("rot_rot_solver");
        const auto sPresPrm = prm.get_child("spres_spres_solver");

        std::function<Linear::DispVector0T<Scalar>()> dispWeightCalc;
        std::function<Linear::RotVectorT<Scalar>()> rotWeightCalc;
        std::function<Linear::SPresVectorT<Scalar>()> sPresWeightCalc;

        if constexpr (isParallel) {
            dispOp0_ = std::make_unique<DispOp>(S_[_0][_0], *comm_);
            dispSolver0_ = std::make_unique<DispSolver>(*dispOp0_,
                                                        *comm_,
                                                        dispPrm,
                                                        dispWeightCalc,
                                                        pressureIdx);

            dispOp1_ = std::make_unique<DispOp>(S_[_1][_1], *comm_);
            dispSolver1_ = std::make_unique<DispSolver>(*dispOp1_,
                                                        *comm_,
                                                        dispPrm,
                                                        dispWeightCalc,
                                                        pressureIdx);

            dispOp2_ = std::make_unique<DispOp>(S_[_2][_2], *comm_);
            dispSolver2_ = std::make_unique<DispSolver>(*dispOp2_,
                                                        *comm_,
                                                        dispPrm,
                                                        dispWeightCalc,
                                                        pressureIdx);

            rotOp_ = std::make_unique<RotOp>(S_[_3][_3], *comm_);
            rotSolver_ = std::make_unique<RotSolver>(*rotOp_,
                                                     *comm_,
                                                     rotPrm,
                                                     rotWeightCalc,
                                                     pressureIdx);

            sPresOp_ = std::make_unique<SPresOp>(S_[_4][_4], *comm_);
            sPresSolver_ = std::make_unique<SPresSolver>(*sPresOp_,
                                                         *comm_,
                                                         sPresPrm,
                                                         sPresWeightCalc,
                                                         pressureIdx);
        } else {
            dispOp0_ = std::make_unique<DispOp>(S_[_0][_0]);
            dispSolver0_ = std::make_unique<DispSolver>(*dispOp0_,
                                                        dispPrm,
                                                        dispWeightCalc,
                                                        pressureIdx);

            dispOp1_ = std::make_unique<DispOp>(S_[_1][_1]);
            dispSolver1_ = std::make_unique<DispSolver>(*dispOp1_,
                                                        dispPrm,
                                                        dispWeightCalc,
                                                        pressureIdx);

            dispOp2_ = std::make_unique<DispOp>(S_[_2][_2]);
            dispSolver2_ = std::make_unique<DispSolver>(*dispOp2_,
                                                        dispPrm,
                                                        dispWeightCalc,
                                                        pressureIdx);

            rotOp_ = std::make_unique<RotOp>(S_[_3][_3]);
            rotSolver_ = std::make_unique<RotSolver>(*rotOp_,
                                                     rotPrm,
                                                     rotWeightCalc,
                                                     pressureIdx);

            sPresOp_ = std::make_unique<SPresOp>(S_[_4][_4]);
            sPresSolver_ = std::make_unique<SPresSolver>(*sPresOp_,
                                                         sPresPrm,
                                                         sPresWeightCalc,
                                                         pressureIdx);
        }
    }

    const Linear::TpsaMatrixView<Scalar>& S_;
    const Comm* comm_{nullptr};

    std::unique_ptr<DispOp> dispOp0_;
    std::unique_ptr<DispOp> dispOp1_;
    std::unique_ptr<DispOp> dispOp2_;
    std::unique_ptr<RotOp> rotOp_;
    std::unique_ptr<SPresOp> sPresOp_;

    std::unique_ptr<DispSolver> dispSolver0_;
    std::unique_ptr<DispSolver> dispSolver1_;
    std::unique_ptr<DispSolver> dispSolver2_;
    std::unique_ptr<RotSolver> rotSolver_;
    std::unique_ptr<SPresSolver> sPresSolver_;
};

} // namespace Opm

#endif // OPM_TPSA_PRECONDITIONER_HPP
