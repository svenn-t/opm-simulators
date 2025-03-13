/*
  Copyright 2020 Equinor ASA.

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

#ifndef OPM_PY_GAS_WATER_SIMULATOR_HEADER_INCLUDED
#define OPM_PY_GAS_WATER_SIMULATOR_HEADER_INCLUDED

#include <python/simulators/PyMainGW.hpp>

#include <opm/simulators/flow/python/PyBaseSimulator.hpp>
#include <opm/simulators/flow/TTagFlowProblemGasWater.hpp>

#include <flow/flow_gaswater.hpp>

#include <memory>

namespace Opm::Pybind {

class PyGasWaterSimulator : public PyBaseSimulator<Opm::Properties::TTag::FlowGasWaterProblem>
{
private:
    using BaseType = PyBaseSimulator<Opm::Properties::TTag::FlowGasWaterProblem>;
    using TypeTag = Opm::Properties::TTag::FlowGasWaterProblem;

public:
    PyGasWaterSimulator(const std::string& deck_filename,
                        const std::vector<std::string>& args)
    : BaseType(deck_filename, args)
    {}

    PyGasWaterSimulator(std::shared_ptr<Opm::Deck> deck,
                        std::shared_ptr<Opm::EclipseState> state,
                        std::shared_ptr<Opm::Schedule> schedule,
                        std::shared_ptr<Opm::SummaryConfig> summary_config,
                        const std::vector<std::string>& args)   
    : BaseType(deck, state, schedule, summary_config, args)
    {}

    int run();
    int stepInit();

private:
    // // This *must* be declared before other pointers
    // // to simulator objects. This in order to deinitialize
    // // MPI at the correct time (ie after the other objects).
    std::unique_ptr<Opm::PyMainGW> main_;
};

} // namespace Opm::Pybind
#endif // OPM_PY_GASWATER_SIMULATOR_HEADER_INCLUDED
