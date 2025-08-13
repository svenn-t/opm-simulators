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

#ifndef OPM_PY_BLACKOIL_SIMULATOR_HEADER_INCLUDED
#define OPM_PY_BLACKOIL_SIMULATOR_HEADER_INCLUDED

#include <opm/simulators/flow/python/PyBaseSimulator.hpp>
#include <opm/simulators/flow/TTagFlowProblemTPFA.hpp>

#include <memory>

namespace Opm::Pybind {

class PyBlackOilSimulator : public PyBaseSimulator<Opm::Properties::TTag::FlowProblemTPFA>
{
private:
    using BaseType = PyBaseSimulator<Opm::Properties::TTag::FlowProblemTPFA>;
    using TypeTag = Opm::Properties::TTag::FlowProblemTPFA;

public:
    PyBlackOilSimulator(const std::string& deck_filename,
                        const std::vector<std::string>& args)
    : BaseType(deck_filename, args)
    {}

    PyBlackOilSimulator(std::shared_ptr<Opm::Deck> deck,
                        std::shared_ptr<Opm::EclipseState> state,
                        std::shared_ptr<Opm::Schedule> schedule,
                        std::shared_ptr<Opm::SummaryConfig> summary_config,
                        const std::vector<std::string>& args)
    : BaseType(deck, state, schedule, summary_config, args)
    {}

};

} // namespace Opm::Pybind
#endif // OPM_PY_BLACKOIL_SIMULATOR_HEADER_INCLUDED
