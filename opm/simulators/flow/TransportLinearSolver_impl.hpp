// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
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
/**
 * \file
 *
 * \copydoc Opm::TransportLinearSolver
 */
#ifndef OPM_TRANSPORT_LINEAR_SOLVER_IMPL_HPP
#define OPM_TRANSPORT_LINEAR_SOLVER_IMPL_HPP

// Improve IDE experience
#ifndef OPM_TRANSPORT_LINEAR_SOLVER_HPP
#include <config.h>
#include <opm/simulators/flow/TransportLinearSolver.hpp>
#endif

#include <dune/istl/operators.hh>
#include <dune/istl/schwarz.hh>

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/TimingMacros.hpp>

#include <opm/grid/CpGrid.hpp>

#include <opm/simulators/linalg/FlexibleSolver.hpp>

#include <stdexcept>
#include <string>
#include <tuple>

namespace Opm {

namespace detail {

#if HAVE_MPI
template <class Matrix, class Vector>
using TransportParOperator =
    Dune::OverlappingSchwarzOperator<Matrix, Vector, Vector,
                                     Dune::OwnerOverlapCopyCommunication<int, int>>;

template <class Matrix, class Vector, class Grid>
std::tuple<std::unique_ptr<TransportParOperator<Matrix, Vector>>,
           std::unique_ptr<Dune::FlexibleSolver<TransportParOperator<Matrix, Vector>>>>
createParallelTransportSolver(const Grid&, const Matrix&, const PropertyTree&)
{
    OPM_THROW(std::logic_error, "Grid not supported for parallel tracer-like transport solvers.");
}

template <class Matrix, class Vector>
std::tuple<std::unique_ptr<TransportParOperator<Matrix, Vector>>,
           std::unique_ptr<Dune::FlexibleSolver<TransportParOperator<Matrix, Vector>>>>
createParallelTransportSolver(const Dune::CpGrid& grid, const Matrix& M, const PropertyTree& prm)
{
    using Operator = TransportParOperator<Matrix, Vector>;
    using Solver = Dune::FlexibleSolver<Operator>;
    const auto& cellComm = grid.cellCommunication();
    auto op = std::make_unique<Operator>(M, cellComm);
    auto dummyWeights = []() { return Vector(); };
    auto solver = std::make_unique<Solver>(*op, cellComm, prm, dummyWeights, 0);
    return {std::move(op), std::move(solver)};
}
#endif

} // namespace detail

template <class Grid, class Matrix, class Vector>
TransportLinearSolver<Grid, Matrix, Vector>::
TransportLinearSolver(const Grid& grid, const PropertyTree& prm)
    : grid_(grid)
    , prm_(prm)
{
}

template <class Grid, class Matrix, class Vector>
PropertyTree TransportLinearSolver<Grid, Matrix, Vector>::
defaultParameters()
{
    PropertyTree prm;
    prm.put("maxiter", 100);
    prm.put("tol", 1e-2);
    prm.put("verbosity", 0);
    prm.put("solver", std::string("bicgstab"));
    prm.put("preconditioner.type", std::string("paroverilu0"));
    return prm;
}

template <class Grid, class Matrix, class Vector>
void TransportLinearSolver<Grid, Matrix, Vector>::
prepare_(const Matrix& M)
{
    if (linearSolver_ && matrix_ == &M) {
        // Same matrix with new values
        pre_->update();
        return;
    }

    matrix_ = &M;
#if HAVE_MPI
    if (grid_.comm().size() > 1) {
        auto [op, solver] = detail::createParallelTransportSolver<Matrix, Vector>(grid_, M, prm_);
        pre_ = &solver->preconditioner();
        op_ = std::move(op);
        linearSolver_ = std::move(solver);
        return;
    }
#endif
    using SeqOperator = Dune::MatrixAdapter<Matrix, Vector, Vector>;
    using SeqSolver = Dune::FlexibleSolver<SeqOperator>;
    auto op = std::make_unique<SeqOperator>(M);
    auto dummyWeights = []() { return Vector(); };
    auto solver = std::make_unique<SeqSolver>(*op, prm_, dummyWeights, 0);
    pre_ = &solver->preconditioner();
    op_ = std::move(op);
    linearSolver_ = std::move(solver);
}

template <class Grid, class Matrix, class Vector>
bool TransportLinearSolver<Grid, Matrix, Vector>::
apply_(Vector& x, Vector& b)
{
    Dune::InverseOperatorResult result;
    linearSolver_->apply(x, b, result);
    return result.converged;
}

template <class Grid, class Matrix, class Vector>
bool TransportLinearSolver<Grid, Matrix, Vector>::
solve(const Matrix& M, Vector& x, Vector& b)
{
    prepare_(M);
    return apply_(x, b);
}

template <class Grid, class Matrix, class Vector>
bool TransportLinearSolver<Grid, Matrix, Vector>::
solve(const Matrix& M, std::vector<Vector>& x, std::vector<Vector>& b)
{
    OPM_TIMEBLOCK(transportLinearSolve);
    bool converged = true;
    bool prepared = false;
    for (std::size_t rhsIdx = 0; rhsIdx < b.size(); ++rhsIdx) {
        x[rhsIdx] = 0.0;
        if (grid_.comm().max(b[rhsIdx].infinity_norm()) == 0.0) {
            continue;
        }
        if (!prepared) {
            prepare_(M);
            prepared = true;
        }
        converged = apply_(x[rhsIdx], b[rhsIdx]) && converged;
    }
    return converged;
}

} // namespace Opm

#endif // OPM_TRANSPORT_LINEAR_SOLVER_IMPL_HPP
