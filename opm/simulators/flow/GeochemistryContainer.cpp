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

#include <config.h>

#include <opm/input/eclipse/EclipseState/SpeciesConfig.hpp>

#include <opm/material/fluidsystems/BlackOilDefaultFluidSystemIndices.hpp>
#include <opm/material/fluidsystems/BlackOilFluidSystem.hpp>
#include <opm/material/fluidsystems/GenericOilGasWaterFluidSystem.hpp>

#include <opm/simulators/flow/GeochemistryContainer.hpp>

#include <opm/output/data/Solution.hpp>

#include <algorithm>
#include <utility>

namespace Opm {

template<class Scalar>
void GeochemistryContainer<Scalar>::
allocate(const unsigned bufferSize,
         const SpeciesConfig& species)
{
    if (!species.empty()) {
        allocated_ = true;
        speciesConcentrations_.resize(species.size());
        for (std::size_t idx = 0; idx < species.size(); ++idx) {
            speciesConcentrations_[idx].resize(bufferSize);
        }
        pH_.resize(bufferSize, 0.0);
    }
}

template<class Scalar>
void GeochemistryContainer<Scalar>::
assignSpeciesConcentrations(const unsigned globalDofIdx,
                            const AssignFunction& concentration)
{
    std::for_each(
        speciesConcentrations_.begin(), speciesConcentrations_.end(),
        [globalDofIdx, idx = 0, &concentration](auto& single_species) mutable
        {
            if (!single_species.empty()) {
                single_species[globalDofIdx] = concentration(idx);
            }
            ++idx;
        }
    );
}

template<class Scalar>
void GeochemistryContainer<Scalar>::
assignPH(const unsigned globalDofIdx, const Scalar ph)
{
    pH_[globalDofIdx] = ph;
}

template<class Scalar>
void GeochemistryContainer<Scalar>::
outputRestart(data::Solution& sol,
              const SpeciesConfig& species)
{
    if (!allocated_) {
        return;
    }

    // Output all species concentrations
    std::for_each(
        species.begin(), species.end(),
        [idx = 0, &sol, this](const auto& single_species) mutable
        {
            sol.insert(single_species.name,
                       UnitSystem::measure::identity,
                       std::move(speciesConcentrations_[idx]),
                       data::TargetType::RESTART_OPM_EXTENDED
                    );
            ++idx;
        }
    );
    
    // Output other geochemistry vectors
    using DataEntry =
        std::tuple<std::string, UnitSystem::measure, std::vector<Scalar>&>;
    auto solutionVectors = std::array {
        DataEntry{"PH",  UnitSystem::measure::identity, pH_}
    };
    std::for_each(
        solutionVectors.begin(), solutionVectors.end(),
        [&sol](auto& entry)
        {
            if (!std::get<2>(entry).empty()) {
                sol.insert(
                    std::get<std::string>(entry),
                    std::get<UnitSystem::measure>(entry),
                    std::move(std::get<2>(entry)),
                    data::TargetType::RESTART_OPM_EXTENDED
                );
            }
        }
    );

    allocated_ = false;
}

template class GeochemistryContainer<double>;

#if FLOW_INSTANTIATE_FLOAT
template class GeochemistryContainer<float>;
#endif

}  // namespace Opm

