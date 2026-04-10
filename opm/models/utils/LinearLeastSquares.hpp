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
#ifndef LINEAR_LEAST_SQUARES_HPP
#define LINEAR_LEAST_SQUARES_HPP

#include <dune/istl/matrix.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/umfpack.hh>


namespace Opm
{

template <class Scalar>
class LinearLeastSquares
{
    using Matrix = Dune::Matrix<Scalar>;
    using Vector = Dune::BlockVector<Scalar>;

public:
    LinearLeastSquares(const Matrix& X, const Vector& y)
        : X_(X), y_(y)
    {
    }

    void solve()
    {
        // Calculate coefficients by solving the normal equations: (X^T*X)*beta = X^T*y
        solveNormalEquations_();
    }

    const Vector& beta() const
    {
        return beta_;
    }

    Scalar operator()(const Vector& x) const
    {
        assert(x.N() == beta_.N());
        // Scalar sum = 0.0;
        // for (std::size_t i = 0; i < x.N(); ++i) {
        //     sum += x[i] * beta_[i];
        // }
        // return sum;
        return beta_ * x;
    }

    Scalar residualSumOfSquares() const
    {
        Vector r(y_);
        X_.mmv(beta_, r);
        return r.two_norm2();
    }

    Scalar explainedSumOfSquares() const
    {
        Scalar ymean = 0.0;
        for (size_t i = 0; i < y_.N(); ++i) {
            ymean += y_[i];
        }
        ymean /= y_.N();

        Vector r(y_.N());
        r = ymean;
        X_.mmv(beta_, r);

        return r.two_norm2();
    }

    Scalar totalSumOfSquares() const
    {
        Scalar ymean = 0.0;
        for (size_t i = 0; i < y_.N(); ++i) {
            ymean += y_[i];
        }
        ymean /= y_.N();

        Vector r(y_.N());
        r = ymean;
        r -= y_;

        return r.two_norm2();
    }

    Scalar RSquared() const
    {
        return explainedSumOfSquares() / totalSumOfSquares();
        // return 1.0 - sumOfSquaredResiduals() / totalSumOfSquares();
    }

private:
    void solveNormalEquations_()
    {
        // Right-hand side, yhat = X^T*y
        Vector yhat(X_.M());
        X_.mtv(y_, yhat);

        // Normal matrix Xhat = X^T*X
        Matrix Xhat = X_.transpose() * X_;

        // Solve normal equations beta = Xhat^{-1}*yhat using UMFPack direct solver
        Dune::UMFPack<Matrix> solver(Xhat, 0);
        beta_.resize(yhat.N());
        beta_ = 0.0;
        Dune::InverseOperatorResult res;
        solver.apply(beta_, yhat, res);
    }

    Matrix X_{};
    Vector y_{};
    Vector beta_{};
};

} // Opm

#endif // LINEAR_LEAST_SQUARES_HPP
