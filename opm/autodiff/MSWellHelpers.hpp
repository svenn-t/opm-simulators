/*
  Copyright 2017 SINTEF Digital, Mathematics and Cybernetics.
  Copyright 2017 Statoil ASA.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef OPM_MSWELLHELPERS_HEADER_INCLUDED
#define OPM_MSWELLHELPERS_HEADER_INCLUDED

#include <opm/common/ErrorMacros.hpp>
// #include <dune/istl/solvers.hh>
#include <dune/istl/umfpack.hh>
#include <cmath>

namespace Opm {

namespace mswellhelpers
{

    // obtain y = D^-1 * x
    template <typename MatrixType, typename VectorType>
    VectorType
    invDX(const MatrixType& D, VectorType x)
    {
        // the function will change the value of x, so we should not use reference of x here.

        // TODO: store some of the following information to avoid to call it again and again for
        // efficiency improvement.
        // Bassically, only the solve / apply step is different.

        VectorType y(x.size());
        y = 0.;

        /* Dune::MatrixAdapter<MatrixType, VectorType, VectorType> linearOperator(D);

        // Sequential incomplete LU decomposition as the preconditioner
        Dune::SeqILU0<MatrixType, VectorType, VectorType> preconditioner(D, 1.0);
        // Dune::SeqILUn<MatrixType, VectorType, VectorType> preconditioner(D, 1, 0.92);
        // Dune::SeqGS<MatrixType, VectorType, VectorType> preconditioner(D, 1, 1);
        // Dune::SeqJac<MatrixType, VectorType, VectorType> preconditioner(D, 1, 1);

        // Preconditioned BICGSTAB solver
        Dune::BiCGSTABSolver<VectorType> linsolver(linearOperator,
                                                   preconditioner,
                                                   // 1.e-8, // desired residual reduction factor
                                                   // 1.e-6, // desired residual reduction factor
                                                   1.e-4, // desired residual reduction factor
                                                   150, // maximum number of iterations
                                                   0); // verbosity of the solver */

        Dune::UMFPack<MatrixType> linsolver(D, 0);

        // Object storing some statistics about the solving process
        Dune::InverseOperatorResult res;

        // Solve
        linsolver.apply(y, x, res);

        if ( !res.converged ) {
            OPM_THROW(Opm::NumericalProblem, "the invDX does not get converged! ");
        }

        return y;
    }





    static double haalandFormular(const double re, const double diameter, const double roughness)
    {
        const double value = -3.6 * std::log10(6.9 / re + std::pow(roughness / (3.7 * diameter), 10. / 9.) );

        // sqrt(1/f) should be non-positive
        assert(value >= 0.0);

        return 1. / (value * value);
    }





    static double calculateFrictionFactor(const double area, const double diameter,
                                          const double w, const double roughness, const double mu)
    {

        double f = 0.;
        // Reynolds number
        const double re = std::abs(diameter * w / (area * mu));

        assert(re > 0.0);

        const double re_value1 = 200.;
        const double re_value2 = 4000.;

        if (re < re_value1) {
            f = 16. / re;
        } else if (re > re_value2){
            f = haalandFormular(re, diameter, roughness);
        } else { // in between
            const double f1 = 16. / re_value1;
            const double f2 = haalandFormular(re_value2, diameter, roughness);

            f = (f2 - f1) / (re_value2 - re_value1) * (re - re_value1) + f1;
        }
        return f;
    }






    // calculating the friction pressure loss
    // l is the segment length
    // area is the segment cross area
    // diameter is the segment inner diameter
    // w is mass flow rate through the segment
    // density is density
    // roughness is the absolute roughness
    // mu is the average phase viscosity
    template <typename ValueType>
    ValueType frictionPressureLoss(const double l, const double diameter, const double area, const double roughness,
                                   const ValueType& density, const ValueType& w, const ValueType& mu)
    {
        const double f = calculateFrictionFactor(area, diameter, w.value(), roughness, mu.value());
        // TODO: a factor of 2 needs to be here based on the dimensional analysis
        return 2. * f * l * w * w / (area * area * diameter * density);
    }





    template <typename ValueType>
    ValueType velocityHead(const double area, const ValueType& mass_rate, const ValueType& density)
    {
        return (0.5 * mass_rate * mass_rate / (area * area * density));
    }


} // namespace mswellhelpers

}

#endif
