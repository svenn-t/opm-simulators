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
#ifndef SYSTEM_PRECONDITIONER_TPSA_HPP
#define SYSTEM_PRECONDITIONER_TPSA_HPP

#include <dune/istl/paamg/pinfo.hh>

#include <opm/simulators/linalg/FlexibleSolver.hpp>
#include "opm/simulators/linalg/ParallelOverlappingILU0.hpp"
#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include "MultiComm.hpp"
#include "SystemTypes.hpp"

namespace Opm
{
// Operator/comm types
template <typename Scalar>
using SeqDispDisp0OperatorT = Dune::MatrixAdapter<typename DispDispMatrix00T<Scalar>::IstlMatrix,
                                                  DispVector0T<Scalar>,
                                                  DispVector0T<Scalar> >;
template <typename Scalar>
using SeqDispDisp1OperatorT = Dune::MatrixAdapter<typename DispDispMatrix11T<Scalar>::IstlMatrix,
                                                  DispVector1T<Scalar>,
                                                  DispVector1T<Scalar> >;
template <typename Scalar>
using SeqDispDisp2OperatorT = Dune::MatrixAdapter<typename DispDispMatrix22T<Scalar>::IstlMatrix,
                                                  DispVector2T<Scalar>,
                                                  DispVector2T<Scalar> >;

template <typename Scalar>
using SeqRotRotOperatorT = Dune::MatrixAdapter<typename RotRotMatrixT<Scalar>::IstlMatrix,
                                               RotVectorT<Scalar>,
                                               RotVectorT<Scalar> >;
template <typename Scalar>
using SeqSPresSPresOperatorT = Dune::MatrixAdapter<typename SPresSPresMatrixT<Scalar>::IstlMatrix,
                                                   SPresVectorT<Scalar>,
                                                   SPresVectorT<Scalar> >;

#if HAVE_MPI
using ParComm = Dune::OwnerOverlapCopyCommunication<int, int>;

template <typename Scalar>
using ParDispDisp0OperatorT =
Dune::OverlappingSchwarzOperator<typename DispDispMatrix00T<Scalar>::IstlMatrix,
                                 DispVector0T<Scalar>,
                                 DispVector0T<Scalar>,
                                 ParComm>;
template <typename Scalar>
using ParDispDisp1OperatorT =
Dune::OverlappingSchwarzOperator<typename DispDispMatrix11T<Scalar>::IstlMatrix,
                                 DispVector1T<Scalar>,
                                 DispVector1T<Scalar>,
                                 ParComm>;
template <typename Scalar>
using ParDispDisp2OperatorT =
Dune::OverlappingSchwarzOperator<typename DispDispMatrix22T<Scalar>::IstlMatrix,
                                 DispVector2T<Scalar>,
                                 DispVector2T<Scalar>,
                                 ParComm>;

template <typename Scalar>
using ParRotRotOperatorT =
    Dune::OverlappingSchwarzOperator<typename RotRotMatrixT<Scalar>::IstlMatrix,
                                     RotVectorT<Scalar>,
                                     RotVectorT<Scalar>,
                                     ParComm>;

template <typename Scalar>
using ParSPresSPresOperatorT =
    Dune::OverlappingSchwarzOperator<typename SPresSPresMatrixT<Scalar>::IstlMatrix,
                                     SPresVectorT<Scalar>,
                                     SPresVectorT<Scalar>,
                                     ParComm>;
#endif

template <class Scalar, class DD00op, class DD11op, class DD22op, class RRop, class SSop,
          class Comm = Dune::Amg::SequentialInformation>
class SystemPreconditionerTPSA : public Dune::PreconditionerWithUpdate<SystemVectorT<Scalar>,
                                                                       SystemVectorT<Scalar> >
{
    using DispDisp00FlexibleSolver = Dune::FlexibleSolver<DD00op>;
    using DispDisp11FlexibleSolver = Dune::FlexibleSolver<DD11op>;
    using DispDisp22FlexibleSolver = Dune::FlexibleSolver<DD22op>;
    using RotRotFlexibleSolver = Dune::FlexibleSolver<RRop>;
    using SPresSPresFlexibleSolver = Dune::FlexibleSolver<SSop>;

public:
    static constexpr bool isParallel = !std::is_same_v<Comm, Dune::Amg::SequentialInformation>;
    static constexpr auto _0 = Dune::Indices::_0;
    static constexpr auto _1 = Dune::Indices::_1;
    static constexpr auto _2 = Dune::Indices::_2;
    static constexpr auto _3 = Dune::Indices::_3;
    static constexpr auto _4 = Dune::Indices::_4;
    static constexpr int pressureIdx = 0;

    // Sequential constructor (enabled only for non-parallel specializations).
    template <bool P = isParallel, std::enable_if_t<!P, int> = 0>
    SystemPreconditionerTPSA(const SystemMatrixT<Scalar>& S, const PropertyTree& prm)
        : S_{S}
    {
        initSubSolvers(prm);
    }

    // Parallel constructor
    template <bool P = isParallel, std::enable_if_t<P, int> = 0>
    SystemPreconditionerTPSA(const SystemMatrixT<Scalar>& S,
                             const PropertyTree& prm,
                             const Comm& comm)
        : S_{S}
        , comm_(&comm)
    {
        initSubSolvers(prm);
    }

    void pre(SystemVectorT<Scalar>&, SystemVectorT<Scalar>&) override
    {
    }
    void post(SystemVectorT<Scalar>&) override
    {
    }

    Dune::SolverCategory::Category category() const override
    {
        if constexpr (isParallel)
            return Dune::SolverCategory::overlapping;
        else
            return Dune::SolverCategory::sequential;
    }

    void update() override
    {
        dispDisp00Solver_->preconditioner().update();
        dispDisp11Solver_->preconditioner().update();
        dispDisp22Solver_->preconditioner().update();
        rotRotSolver_->preconditioner().update();
        sPresSpresSolver_->preconditioner().update();
    }

    bool hasPerfectUpdate() const override
    {
        return true;
    }

    void apply(SystemVectorT<Scalar>& v, const SystemVectorT<Scalar>& d) override
    {
        Dune::InverseOperatorResult result0;
        Dune::InverseOperatorResult result1;
        Dune::InverseOperatorResult result2;
        Dune::InverseOperatorResult result3;
        Dune::InverseOperatorResult result4;
        auto d0 = d[_0];
        auto d1 = d[_1];
        auto d2 = d[_2];
        auto d3 = d[_3];
        auto d4 = d[_4];

        dispDisp00Solver_->apply(v[_0], d0, result0);
        dispDisp11Solver_->apply(v[_1], d1, result1);
        dispDisp22Solver_->apply(v[_2], d2, result2);
        // if constexpr (isParallel) {
        //     comm_->copyOwnerToAll(v[_0], v[_0]);
        // }

        S_[_3][_0].istlMatrix().mmv(v[_0], d3);
        S_[_3][_1].istlMatrix().mmv(v[_1], d3);
        S_[_3][_2].istlMatrix().mmv(v[_2], d3);
        rotRotSolver_->apply(v[_3], d3, result3);
        // if constexpr (isParallel) {
        //     comm_->project(d1);
        // }


        S_[_4][_0].istlMatrix().mmv(v[_0], d4);
        S_[_4][_1].istlMatrix().mmv(v[_1], d4);
        S_[_4][_2].istlMatrix().mmv(v[_2], d4);
        sPresSpresSolver_->apply(v[_4], d4, result4);
        // if constexpr (isParallel) {
        //     comm_->project(d2);
        // }
    }

private:
    void initSubSolvers(const PropertyTree& prm)
    {
        // Displacement-displacement preconditioners
        auto dispPrm = prm.get_child("disp_disp_solver");
        std::function<DispVector0T<Scalar>()> dispWeightCalc;
        if constexpr (isParallel) {
            dispDisp00Op_ = std::make_unique<DD00op>(S_[_0][_0].istlMatrix(), *comm_);
            dispDisp00Solver_ =
                std::make_unique<DispDisp00FlexibleSolver>(*dispDisp00Op_,
                                                           *comm_,
                                                           dispPrm,
                                                           dispWeightCalc,
                                                           pressureIdx);

            dispDisp11Op_ = std::make_unique<DD11op>(S_[_1][_1].istlMatrix(), *comm_);
            dispDisp11Solver_ =
                std::make_unique<DispDisp11FlexibleSolver>(*dispDisp11Op_,
                                                           *comm_,
                                                           dispPrm,
                                                           dispWeightCalc,
                                                           pressureIdx);

            dispDisp22Op_ = std::make_unique<DD22op>(S_[_2][_2].istlMatrix(), *comm_);
            dispDisp22Solver_ =
                std::make_unique<DispDisp22FlexibleSolver>(*dispDisp22Op_,
                                                           *comm_,
                                                           dispPrm,
                                                           dispWeightCalc,
                                                           pressureIdx);
        } else {
            dispDisp00Op_ = std::make_unique<DD00op>(S_[_0][_0].istlMatrix());
            dispDisp00Solver_ =
                std::make_unique<DispDisp00FlexibleSolver>(*dispDisp00Op_,
                                                           dispPrm,
                                                           dispWeightCalc,
                                                           pressureIdx);

            dispDisp11Op_ = std::make_unique<DD11op>(S_[_1][_1].istlMatrix());
            dispDisp11Solver_ =
                std::make_unique<DispDisp11FlexibleSolver>(*dispDisp11Op_,
                                                           dispPrm,
                                                           dispWeightCalc,
                                                           pressureIdx);

            dispDisp22Op_ = std::make_unique<DD22op>(S_[_2][_2].istlMatrix());
            dispDisp22Solver_ =
                std::make_unique<DispDisp22FlexibleSolver>(*dispDisp22Op_,
                                                           dispPrm,
                                                           dispWeightCalc,
                                                           pressureIdx);
        }

        // Rotation-rotation preconditioner
        auto rotPrm = prm.get_child("rot_rot_solver");
        std::function<RotVectorT<Scalar>()> rotWeightCalc;
        if constexpr (isParallel) {
            rotRotOp_ = std::make_unique<RRop>(S_[_3][_3].istlMatrix(), *comm_);
            rotRotSolver_ =
                std::make_unique<RotRotFlexibleSolver>(*rotRotOp_,
                                                        *comm_,
                                                        rotPrm,
                                                        rotWeightCalc,
                                                        pressureIdx);
        } else {
            rotRotOp_ = std::make_unique<RRop>(S_[_3][_3].istlMatrix());
            rotRotSolver_ =
                std::make_unique<RotRotFlexibleSolver>(*rotRotOp_,
                                                       rotPrm,
                                                       rotWeightCalc,
                                                       pressureIdx);
        }

        // Solid pressure-solid pressure preconditioner
        auto sPresPrm = prm.get_child("spres_spres_solver");
        std::function<SPresVectorT<Scalar>()> sPresWeightCalc;
        if constexpr (isParallel) {
            sPresSpresOp_ = std::make_unique<SSop>(S_[_4][_4].istlMatrix(), *comm_);
            sPresSpresSolver_ =
                std::make_unique<SPresSPresFlexibleSolver>(*sPresSpresOp_,
                                                           *comm_,
                                                           sPresPrm,
                                                           sPresWeightCalc,
                                                           pressureIdx);
        } else {
            sPresSpresOp_ = std::make_unique<SSop>(S_[_4][_4].istlMatrix());
            sPresSpresSolver_ =
                std::make_unique<SPresSPresFlexibleSolver>(*sPresSpresOp_,
                                                           sPresPrm,
                                                           sPresWeightCalc,
                                                           pressureIdx);
        }
    }

    const SystemMatrixT<Scalar>& S_;
    const Comm* comm_ = nullptr;

    std::unique_ptr<DD00op> dispDisp00Op_;
    std::unique_ptr<DD11op> dispDisp11Op_;
    std::unique_ptr<DD22op> dispDisp22Op_;
    std::unique_ptr<DispDisp00FlexibleSolver> dispDisp00Solver_;
    std::unique_ptr<DispDisp11FlexibleSolver> dispDisp11Solver_;
    std::unique_ptr<DispDisp22FlexibleSolver> dispDisp22Solver_;
    std::unique_ptr<RRop> rotRotOp_;
    std::unique_ptr<RotRotFlexibleSolver> rotRotSolver_;
    std::unique_ptr<SSop> sPresSpresOp_;
    std::unique_ptr<SPresSPresFlexibleSolver> sPresSpresSolver_;
};
}

#endif //SYSTEM_PRECONDITIONER_TPSA_HPP
