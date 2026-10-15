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
#ifndef OPM_TRANSPORT_LINEAR_SOLVER_HPP
#define OPM_TRANSPORT_LINEAR_SOLVER_HPP

#include <dune/istl/operators.hh>
#include <dune/istl/solver.hh>

#include <opm/simulators/linalg/PreconditionerWithUpdate.hpp>
#include <opm/simulators/linalg/PropertyTree.hpp>

#include <memory>
#include <vector>

namespace Opm {

/*!
 * \brief Linear solver for the sequentially solved, tracer-like transport
 *        equations (see TransportLinearSystem).
 *
 * The solver is set up for a matrix on the first call to solve(). Later calls
 * only update the preconditioner, so the matrix must keep its address and
 * sparsity pattern between calls.
 *
 * The implementation is in TransportLinearSolver_impl.hpp, which is included
 * by the translation units instantiating the models using it.
 */
template <class Grid, class Matrix, class Vector>
class TransportLinearSolver
{
public:
    TransportLinearSolver(const Grid& grid, const PropertyTree& prm);

    /*!
     * \brief Solve M x = b. The right-hand side b is overwritten.
     *
     * \return Whether the linear solver converged.
     */
    bool solve(const Matrix& M, Vector& x, Vector& b);

    /*!
     * \brief Solve M x[i] = b[i] for several right-hand sides with the same matrix.
     *
     * Systems with a zero right-hand side are not passed to the solver.
     * The right-hand sides are overwritten.
     *
     * \return Whether the linear solver converged for all right-hand sides.
     */
    bool solve(const Matrix& M, std::vector<Vector>& x, std::vector<Vector>& b);

    /*!
     * \brief Default parameters: BiCGStab with ILU0 (overlapping in parallel).
     */
    static PropertyTree defaultParameters();

private:
    void prepare_(const Matrix& M);
    bool apply_(Vector& x, Vector& b);

    using AbstractSolverType = Dune::InverseOperator<Vector, Vector>;
    using AbstractOperatorType = Dune::AssembledLinearOperator<Matrix, Vector, Vector>;
    using AbstractPreconditionerType = Dune::PreconditionerWithUpdate<Vector, Vector>;

    const Grid& grid_;
    PropertyTree prm_;
    const Matrix* matrix_ = nullptr;
    std::unique_ptr<AbstractOperatorType> op_;
    std::unique_ptr<AbstractSolverType> linearSolver_;
    AbstractPreconditionerType* pre_ = nullptr;
};

} // namespace Opm

#endif // OPM_TRANSPORT_LINEAR_SOLVER_HPP
