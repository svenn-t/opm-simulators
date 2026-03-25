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
using SeqDispDispOperatorT = Dune::MatrixAdapter<typename DispDispMatrixT<Scalar>::IstlMatrix,
                                                 DispVectorT<Scalar>,
                                                 DispVectorT<Scalar> >;

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
using ParDispDispOperatorT =
    Dune::OverlappingSchwarzOperator<typename DispDispMatrixT<Scalar>::IstlMatrix,
                                     DispVectorT<Scalar>,
                                     DispVectorT<Scalar>,
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

template <class Scalar, class DDop, class RRop, class SSop, class Comm =
          Dune::Amg::SequentialInformation>
class SystemPreconditionerTPSA : public Dune::PreconditionerWithUpdate<SystemVectorT<Scalar>,
                                                                       SystemVectorT<Scalar> >
{
    using DispDispFlexibleSolver = Dune::FlexibleSolver<DDop>;
    using RotRotFlexibleSolver = Dune::FlexibleSolver<RRop>;
    using SPresSPresFlexibleSolver = Dune::FlexibleSolver<SSop>;
public:
    static constexpr bool isParallel = !std::is_same_v<Comm, Dune::Amg::SequentialInformation>;
    static constexpr auto _0 = Dune::Indices::_0;
    static constexpr auto _1 = Dune::Indices::_1;
    static constexpr auto _2 = Dune::Indices::_2;
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
        dispDispSolver_->preconditioner().update();
        rotRotSolver_->preconditioner().update();
        sPresSpresSolver_->preconditioner().update();
    }

    bool hasPerfectUpdate() const override
    {
        return true;
    }

    void apply(SystemVectorT<Scalar>& v, const SystemVectorT<Scalar>& d) override
    {
        Dune::InverseOperatorResult result1;
        Dune::InverseOperatorResult result2;
        Dune::InverseOperatorResult result3;
        auto d0 = d[_0];
        auto d1 = d[_1];
        auto d2 = d[_2];

        dispDispSolver_->apply(v[_0], d0, result1);
        if constexpr (isParallel) {
            comm_->copyOwnerToAll(v[_0], v[_0]);
        }

        S_[_1][_0].istlMatrix().mmv(v[_0], d1);
        if constexpr (isParallel) {
            comm_->project(d1);
        }
        rotRotSolver_->apply(v[_1], d1, result2);

        S_[_2][_0].istlMatrix().mmv(v[_0], d2);
        if constexpr (isParallel) {
            comm_->project(d2);
        }

        sPresSpresSolver_->apply(v[_2], d2, result3);
    }

private:
    void initSubSolvers(const PropertyTree& prm)
    {
        // Displacement-displacement preconditioner
        auto dispPrm = prm.get_child("disp_disp_solver");
        std::function<DispVectorT<Scalar>()> dispWeightCalc;
        if constexpr (isParallel) {
            dispDispOp_ = std::make_unique<DDop>(S_[_0][_0].istlMatrix(), *comm_);
            dispDispSolver_ =
                std::make_unique<DispDispFlexibleSolver>(*dispDispOp_,
                                                         *comm_,
                                                         dispPrm,
                                                         dispWeightCalc,
                                                         pressureIdx);
        } else {
            dispDispOp_ = std::make_unique<DDop>(S_[_0][_0].istlMatrix());
            dispDispSolver_ =
                std::make_unique<DispDispFlexibleSolver>(*dispDispOp_,
                                                         dispPrm,
                                                         dispWeightCalc,
                                                         pressureIdx);
        }

        // Rotation-rotation preconditioner
        auto rotPrm = prm.get_child("rot_rot_solver");
        std::function<DispVectorT<Scalar>()> rotWeightCalc;
        if constexpr (isParallel) {
            rotRotOp_ = std::make_unique<RRop>(S_[_1][_1].istlMatrix(), *comm_);
            rotRotSolver_ =
                std::make_unique<RotRotFlexibleSolver>(*rotRotOp_,
                                                        *comm_,
                                                        rotPrm,
                                                        rotWeightCalc,
                                                        pressureIdx);
        } else {
            rotRotOp_ = std::make_unique<RRop>(S_[_1][_1].istlMatrix());
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
            sPresSpresOp_ = std::make_unique<SSop>(S_[_2][_2].istlMatrix(), *comm_);
            sPresSpresSolver_ =
                std::make_unique<SPresSPresFlexibleSolver>(*sPresSpresOp_,
                                                           *comm_,
                                                           sPresPrm,
                                                           sPresWeightCalc,
                                                           pressureIdx);
        } else {
            sPresSpresOp_ = std::make_unique<SSop>(S_[_2][_2].istlMatrix());
            sPresSpresSolver_ =
                std::make_unique<SPresSPresFlexibleSolver>(*sPresSpresOp_,
                                                           sPresPrm,
                                                           sPresWeightCalc,
                                                           pressureIdx);
        }
    }

    const SystemMatrixT<Scalar>& S_;
    const Comm* comm_ = nullptr;

    std::unique_ptr<DDop> dispDispOp_;
    std::unique_ptr<DispDispFlexibleSolver> dispDispSolver_;
    std::unique_ptr<RRop> rotRotOp_;
    std::unique_ptr<RotRotFlexibleSolver> rotRotSolver_;
    std::unique_ptr<SSop> sPresSpresOp_;
    std::unique_ptr<SPresSPresFlexibleSolver> sPresSpresSolver_;
};
}

#endif //SYSTEM_PRECONDITIONER_TPSA_HPP
