// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2026, NORCE AS

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
#include "config.h"

#include <flow/flow_gaswater_dissolution_solvent_tpsa.hpp>

#include <opm/material/thermal/EnergyModuleType.hpp>

#include <opm/models/blackoil/blackoilconvectivemixingmodule.hh>
#include <opm/models/blackoil/blackoilenergymodules.hh>
#include <opm/models/blackoil/blackoillocalresidualtpfa.hh>
#include <opm/models/blackoil/blackoilsolventmodules.hh>
#include <opm/models/blackoil/blackoiltwophaseindices.hh>

#include <opm/models/discretization/common/tpfalinearizer.hh>

#include <opm/simulators/flow/BlackoilModelProperties.hpp>
#include <opm/simulators/flow/Main.hpp>
#include <opm/simulators/flow/TTagFlowProblemTPSA.hpp>

#include <tuple>


namespace Opm::Properties {

namespace TTag {

struct FlowGasWaterDissolutionSolventProblemTPSA
{ using InheritsFrom = std::tuple<FlowProblem, FlowProblemTpsa>; };

}  // namespace Opm::Properties::TTag

// ///
// Flow related properties
// ///
template<class TypeTag>
struct EnableDiffusion<TypeTag, TTag::FlowGasWaterDissolutionSolventProblemTPSA>
{ static constexpr bool value = false; };

template<class TypeTag>
struct EnableDisgasInWater<TypeTag, TTag::FlowGasWaterDissolutionSolventProblemTPSA>
{ static constexpr bool value = true; };

template<class TypeTag>
struct EnableVapwat<TypeTag, TTag::FlowGasWaterDissolutionSolventProblemTPSA>
{ static constexpr bool value = true; };

template<class TypeTag>
struct EnergyModuleType<TypeTag, TTag::FlowGasWaterDissolutionSolventProblemTPSA>
{ static constexpr EnergyModules value = EnergyModules::ConstantTemperature; };

template<class TypeTag>
struct EnableSolvent<TypeTag, TTag::FlowGasWaterDissolutionSolventProblemTPSA>
{ static constexpr bool value = true; };

//! The indices required by the model
template<class TypeTag>
struct Indices<TypeTag, TTag::FlowGasWaterDissolutionSolventProblemTPSA>
{
private:
    // it is unfortunately not possible to simply use 'TypeTag' here because this leads
    // to cyclic definitions of some properties. if this happens the compiler error
    // messages unfortunately are *really* confusing and not really helpful.
    using BaseTypeTag = TTag::FlowProblem;
    using FluidSystem = GetPropType<BaseTypeTag, Properties::FluidSystem>;
    static constexpr EnergyModules energyModuleType = getPropValue<TypeTag, Properties::EnergyModuleType>();
    static constexpr int numEnergyVars = energyModuleType == EnergyModules::FullyImplicitThermal ? 1 : 0;

public:
    using type = BlackOilTwoPhaseIndices<getPropValue<TypeTag, Properties::EnableSolvent>(),
                                         getPropValue<TypeTag, Properties::EnableExtbo>(),
                                         getPropValue<TypeTag, Properties::EnablePolymer>(),
                                         numEnergyVars,
                                         getPropValue<TypeTag, Properties::EnableFoam>(),
                                         getPropValue<TypeTag, Properties::EnableBrine>(),
                                         /*PVOffset=*/0,
                                         /*disabledCompIdx=*/FluidSystem::oilCompIdx,
                                         getPropValue<TypeTag, Properties::EnableBioeffects>()>;
};

// ///
// TPSA related properties
// ///
template <class TypeTag>
struct EnableMech<TypeTag, TTag::FlowGasWaterDissolutionSolventProblemTPSA>
{ static constexpr bool value = true; };

template <class TypeTag>
struct Problem<TypeTag, TTag::FlowGasWaterDissolutionSolventProblemTPSA>
{ using type = FlowProblemTPSA<TypeTag>; };

template <class TypeTag>
struct NonlinearSystem<TypeTag, TTag::FlowGasWaterDissolutionSolventProblemTPSA>
{ using type = NonlinearSystemBlackOilReservoirTPSA<TypeTag>; };

}  // namespace Opm::Properties

namespace Opm {

// ----------------- Main program -----------------
int flowGasWaterDissolutionSolventTpsaMain(int argc, char** argv, bool outputCout, bool outputFiles)
{
    // we always want to use the default locale, and thus spare us the trouble
    // with incorrect locale settings.
    resetLocale();

    FlowMain<Properties::TTag::FlowGasWaterDissolutionSolventProblemTPSA>
        mainfunc {argc, argv, outputCout, outputFiles};
    return mainfunc.execute();
}

int flowGasWaterDissolutionSolventTpsaMainStandalone(int argc, char** argv)
{
    using TypeTag = Properties::TTag::FlowGasWaterDissolutionSolventProblemTPSA;
    auto mainObject = std::make_unique<Opm::Main>(argc, argv);
    auto ret = mainObject->runStatic<TypeTag>();
    // Destruct mainObject as the destructor calls MPI_Finalize!
    mainObject.reset();
    return ret;
}

}  // namespace Opm
