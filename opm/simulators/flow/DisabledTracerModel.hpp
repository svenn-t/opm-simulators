// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright Equinor ASA 2026

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
 * \copydoc Opm::DisabledTracerModel
 */
#ifndef OPM_DISABLED_TRACER_MODEL_HPP
#define OPM_DISABLED_TRACER_MODEL_HPP

#include <opm/common/ErrorMacros.hpp>

#include <opm/models/common/multiphasebaseproperties.hh>
#include <opm/models/discretization/common/fvbaseproperties.hh>
#include <opm/models/utils/basicproperties.hh>
#include <opm/models/utils/propertysystem.hh>

#include <opm/simulators/flow/GenericTracerModel.hpp>

#include <stdexcept>

namespace Opm {

/*!
 * \ingroup BlackOilSimulator
 *
 * \brief Tracer model without tracer transport, for the flow binaries that assemble the flow
 *        equations with element contexts. TracerModel needs the face data of the TpfaLinearizer.
 *        Decks with tracers are rejected.
 */
template <class TypeTag>
class DisabledTracerModel : public GenericTracerModel<GetPropType<TypeTag, Properties::Grid>,
                                                      GetPropType<TypeTag, Properties::GridView>,
                                                      GetPropType<TypeTag, Properties::DofMapper>,
                                                      GetPropType<TypeTag, Properties::Stencil>,
                                                      GetPropType<TypeTag, Properties::FluidSystem>,
                                                      GetPropType<TypeTag, Properties::Scalar>>
{
    using BaseType = GenericTracerModel<GetPropType<TypeTag, Properties::Grid>,
                                        GetPropType<TypeTag, Properties::GridView>,
                                        GetPropType<TypeTag, Properties::DofMapper>,
                                        GetPropType<TypeTag, Properties::Stencil>,
                                        GetPropType<TypeTag, Properties::FluidSystem>,
                                        GetPropType<TypeTag, Properties::Scalar>>;
    using Simulator = GetPropType<TypeTag, Properties::Simulator>;

public:
    explicit DisabledTracerModel(Simulator& simulator)
        : BaseType(simulator.vanguard().gridView(),
                   simulator.vanguard().eclState(),
                   simulator.vanguard().cartesianIndexMapper(),
                   simulator.model().dofMapper(),
                   simulator.vanguard().cellCentroids())
    { }

    void init(bool /* rst */)
    {
        if (this->numTracers() > 0) {
            OPM_THROW(std::invalid_argument,
                      "Tracers are not supported by this flow binary, which assembles the "
                      "flow equations with element contexts");
        }
    }

    void prepareTracerBatches() {}
    void beginTimeStep() {}
    void endTimeStep() {}

    template <class Restarter>
    void serialize(Restarter&)
    { /* not implemented */ }

    template <class Restarter>
    void deserialize(Restarter&)
    { /* not implemented */ }

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
        serializer(static_cast<BaseType&>(*this));
    }
};

} // namespace Opm

#endif // OPM_DISABLED_TRACER_MODEL_HPP
