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
 * \copydoc Opm::TemperatureModel
 */
#ifndef OPM_GENERIC_TEMPERATURE_MODEL_IMPL_HPP
#define OPM_GENERIC_TEMPERATURE_MODEL_IMPL_HPP

// Improve IDE experience
#ifndef OPM_GENERIC_TEMPERATURE_MODEL_HPP
#include <config.h>
#include <opm/simulators/flow/GenericTemperatureModel.hpp>
#endif

#include <opm/common/OpmLog/OpmLog.hpp>

#include <opm/grid/CpGrid.hpp>

#include <opm/input/eclipse/EclipseState/EclipseState.hpp>
#include <opm/input/eclipse/Schedule/Well/Well.hpp>

#include <opm/models/discretization/ecfv/ecfvstencil.hh>

#include <opm/simulators/flow/TransportLinearSolver_impl.hpp>

#include <memory>
#include <stdexcept>
#include <string>

#include <fmt/format.h>

namespace Opm {

template<class Grid, class GridView, class DofMapper, class Stencil, class FluidSystem, class Scalar>
GenericTemperatureModel<Grid,GridView,DofMapper,Stencil,FluidSystem,Scalar>::
GenericTemperatureModel(const GridView& gridView,
                   const EclipseState& eclState,
                   const CartesianIndexMapper& cartMapper,
                   const DofMapper& dofMapper)
    : gridView_(gridView)
    , eclState_(eclState)
    , cartMapper_(cartMapper)
    , dofMapper_(dofMapper)
    , linearSolver_(gridView.grid(), decltype(linearSolver_)::defaultParameters())
{
}

template<class Grid, class GridView, class DofMapper, class Stencil, class FluidSystem, class Scalar>
void GenericTemperatureModel<Grid,GridView,DofMapper,Stencil,FluidSystem,Scalar>::
doInit(std::size_t numGridDof)
{
    doTemp_ = eclState_.getSimulationConfig().isTemp();
    temperature_.resize(numGridDof);

    if (!doTemp_)
        return;

    energyVector_.resize(numGridDof); 
    maxTempChange_ = Parameters::Get<Parameters::MaxTemperatureChange<Scalar>>();
}

template<class Grid, class GridView, class DofMapper, class Stencil, class FluidSystem, class Scalar>
bool GenericTemperatureModel<Grid,GridView,DofMapper,Stencil,FluidSystem,Scalar>::
linearSolve_(const EnergyMatrix& M, EnergyVector& x, EnergyVector& b)
{
    return linearSolver_.solve(M, x, b);
}

} // namespace Opm

#endif // OPM_GENERIC_TEMPERATURE_MODEL_IMPL_HPP
