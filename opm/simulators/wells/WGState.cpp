/*
  Copyright 2021 Equinor ASA

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

#if HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#include <opm/simulators/wells/WGState.hpp>

#include <opm/simulators/utils/BlackoilPhases.hpp>

namespace Opm {

template<class Scalar>
WGState<Scalar>::WGState(const PhaseUsage& pu) :
    well_state(pu),
    group_state(pu.num_phases),
    well_test_state{}
{}

template<class Scalar>
WGState<Scalar> WGState<Scalar>::
serializationTestObject(const ParallelWellInfo<Scalar>& pinfo)
{
    WGState result(PhaseUsage{});
    result.well_state = WellState<Scalar>::serializationTestObject(pinfo);
    result.group_state = GroupState<Scalar>::serializationTestObject();
    result.well_test_state = WellTestState::serializationTestObject();

    return result;
}

template<class Scalar>
void WGState<Scalar>::wtest_state(std::unique_ptr<WellTestState> wtest_state)
{
    wtest_state->filter_wells( this->well_state.wells() );
    this->well_test_state = std::move(*wtest_state);
}

template<class Scalar>
bool WGState<Scalar>::operator==(const WGState& rhs) const
{
    return this->well_state == rhs.well_state &&
           this->group_state == rhs.group_state &&
           this->well_test_state == rhs.well_test_state;
}

template struct WGState<double>;

#if FLOW_INSTANTIATE_FLOAT
template struct WGState<float>;
#endif

}
