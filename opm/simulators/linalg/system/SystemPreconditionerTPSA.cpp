// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2025 NORCE AS

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
#include "SystemPreconditionerTPSA.hpp"
#include "SystemPreconditionerFactoryTPSA.hpp"

#include <opm/simulators/linalg/FlexibleSolver_impl.hpp>
#include <opm/simulators/linalg/PreconditionerFactory_impl.hpp>

#define INSTANTIATE_SYSTEM_PF_SEQ(T) \
    template class Opm::SystemPreconditionerTPSA<T, \
                                                 Opm::SeqDispDisp0OperatorT<T>,\
                                                 Opm::SeqDispDisp1OperatorT<T>,\
                                                 Opm::SeqDispDisp2OperatorT<T>,\
                                                 Opm::SeqRotRotOperatorT<T>,\
                                                 Opm::SeqSPresSPresOperatorT<T>>; \
    template class Dune::FlexibleSolver<Opm::SystemSeqOpT<T>>; \
    template class Opm::PreconditionerFactory<Opm::SystemSeqOpT<T>, \
                                              Dune::Amg::SequentialInformation>;

#if HAVE_MPI
#define INSTANTIATE_SYSTEM_PF_PAR(T) \
    template class Opm::SystemPreconditionerTPSA<T, \
                                                 Opm::ParDispDisp0OperatorT<T>, \
                                                 Opm::ParDispDisp1OperatorT<T>, \
                                                 Opm::ParDispDisp2OperatorT<T>, \
                                                 Opm::ParRotRotOperatorT<T>, \
                                                 Opm::ParSPresSPresOperatorT<T>, \
                                                 Opm::ParComm>; \
    template class Dune::FlexibleSolver<Opm::SystemParOpT<T>>;                            \
    template Dune::FlexibleSolver<Opm::SystemParOpT<T>>::FlexibleSolver(                   \
        Opm::SystemParOpT<T>& op,                                                         \
        const Opm::SystemComm& comm,                                                      \
        const Opm::PropertyTree& prm,                                                     \
        const std::function<Opm::SystemVectorT<T>()>& weightsCalculator,                  \
        std::size_t pressureIndex); \
    template class Opm::PreconditionerFactory<Opm::SystemParOpT<T>, Opm::SystemComm>; \
    template class Opm::PreconditionerFactory<Opm::SystemParOpT<T>,  \
                                              Dune::Amg::SequentialInformation>;

#define INSTANTIATE_SYSTEM_PF(T) \
    INSTANTIATE_SYSTEM_PF_PAR(T) \
    INSTANTIATE_SYSTEM_PF_SEQ(T)

#else
#define INSTANTIATE_SYSTEM_PF(T) INSTANTIATE_SYSTEM_PF_SEQ(T)
#endif

INSTANTIATE_SYSTEM_PF(double)

#if FLOW_INSTANTIATE_FLOAT
INSTANTIATE_SYSTEM_PF(float)
#endif
