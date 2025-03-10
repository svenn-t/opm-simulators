/*
  Copyright 2025 Equinor ASA.

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

#ifndef TTAG_FLOW_PROBLEM_GASWATER_HPP
#define TTAG_FLOW_PROBLEM_GASWATER_HPP

#include <tuple>

namespace Opm {
namespace Properties {
namespace TTag {

    struct FlowProblem;

    /// Specialised type tag for simulations that can use the customised
    /// assembly process for TPFA discretisation schemes.
    ///
    /// All properties are otherwise the same as for the regular
    /// FlowProblem.
    struct FlowGasWaterProblem {
      using InheritsFrom = std::tuple<FlowProblem>;
  };
}


} // namespace Opm::Properties::TTag
}
#endif // TTAG_FLOW_PROBLEM_TPFA_HPP
