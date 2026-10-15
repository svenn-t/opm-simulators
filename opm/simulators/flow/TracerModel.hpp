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
 * \copydoc Opm::TracerModel
 */
#ifndef OPM_TRACER_MODEL_HPP
#define OPM_TRACER_MODEL_HPP

#include <opm/common/OpmLog/OpmLog.hpp>
#include <opm/common/TimingMacros.hpp>

#include <opm/input/eclipse/EclipseState/Aquifer/AquiferConfig.hpp>
#include <opm/input/eclipse/EclipseState/Grid/FaceDir.hpp>
#include <opm/input/eclipse/Schedule/Well/Well.hpp>
#include <opm/input/eclipse/Schedule/Well/WellConnections.hpp>

#include <opm/models/utils/propertysystem.hh>

#include <opm/simulators/flow/GenericTracerModel.hpp>
#include <opm/simulators/flow/TransportLinearSystem.hpp>
#include <opm/simulators/utils/DeferredLoggingErrorHelpers.hpp>
#include <opm/simulators/utils/gatherDeferredLogger.hpp>
#include <opm/simulators/utils/VectorVectorDataHandle.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

namespace Opm::Properties {

template<class TypeTag, class MyTypeTag>
struct EnableTracerModel {
    using type = UndefinedProperty;
};

} // namespace Opm::Properties

namespace Opm {

/*!
 * \ingroup BlackOilSimulator
 *
 * \brief A class which handles tracers as specified in by ECL
 */
template <class TypeTag>
class TracerModel : public GenericTracerModel<GetPropType<TypeTag, Properties::Grid>,
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
    using GridView = GetPropType<TypeTag, Properties::GridView>;
    using Grid = GetPropType<TypeTag, Properties::Grid>;
    using Scalar = GetPropType<TypeTag, Properties::Scalar>;
    using Stencil = GetPropType<TypeTag, Properties::Stencil>;
    using FluidSystem = GetPropType<TypeTag, Properties::FluidSystem>;
    using Evaluation = GetPropType<TypeTag, Properties::Evaluation>;
    using IntensiveQuantities = GetPropType<TypeTag, Properties::IntensiveQuantities>;
    using ExtensiveQuantities = GetPropType<TypeTag, Properties::ExtensiveQuantities>;

    using TracerMatrix = typename BaseType::TracerMatrix;
    using TracerVector = typename BaseType::TracerVector;
    using TracerVectorSingle = typename BaseType::TracerVectorSingle;

    //! \brief Static data of an interior face, see phaseVolumeFlux_().
    struct FaceInfo
    {
        Scalar dZg{}; //!< Depth difference (interior - exterior) times gravity
        Scalar Vin{}; //!< Bulk volume of the interior cell
        Scalar Vex{}; //!< Bulk volume of the exterior cell
        FaceDir::DirEnum faceDir{FaceDir::DirEnum::Unknown};
    };

    using LinearSystem = TransportLinearSystem<Scalar, 2, FaceInfo>;
    using NeighborInfo = typename LinearSystem::NeighborInfo;
    static_assert(std::is_same_v<typename LinearSystem::Matrix, TracerMatrix>);

    //! \brief Surface volume rate of a tracer phase across a face, out of the cell.
    struct FaceFlux
    {
        Scalar rate{0.0};
        bool upIsInterior{false}; //!< Whether the cell is upstream
    };

    enum { numEq = getPropValue<TypeTag, Properties::NumEq>() };
    enum { numPhases = FluidSystem::numPhases };
    enum { waterPhaseIdx = FluidSystem::waterPhaseIdx };
    enum { oilPhaseIdx = FluidSystem::oilPhaseIdx };
    enum { gasPhaseIdx = FluidSystem::gasPhaseIdx };

public:
    explicit TracerModel(Simulator& simulator)
        : BaseType(simulator.vanguard().gridView(),
                   simulator.vanguard().eclState(),
                   simulator.vanguard().cartesianIndexMapper(),
                   simulator.model().dofMapper(),
                   simulator.vanguard().cellCentroids())
        , simulator_(simulator)
        , tbatch({waterPhaseIdx, oilPhaseIdx, gasPhaseIdx})
        , wat_(tbatch[0])
        , oil_(tbatch[1])
        , gas_(tbatch[2])
    { }


    /*
      The initialization of the tracer model is a three step process:

      1. The init() method is called. This will allocate buffers and initialize
         some phase index stuff. If this is a normal run the initial tracer
         concentrations will be assigned from the TBLK or TVDPF keywords.

      2. [Restart only:] The tracer concentration are read from the restart
         file and the concentrations are applied with repeated calls to the
         setTracerConcentration() method. This is currently done in the
         eclwriter::beginRestart() method.

      3. Internally the tracer model manages the concentrations in "batches" for
         the oil, water and gas tracers respectively. The batches should be
         initialized with the initial concentration, that must be performed
         after the concentration values have been assigned. This is done in
         method prepareTracerBatches() called from eclproblem::finishInit().
    */
    void init(bool rst)
    {
        this->doInit(rst, simulator_.model().numGridDof(),
                     gasPhaseIdx, oilPhaseIdx, waterPhaseIdx);
    }

    void prepareTracerBatches()
    {
        DeferredLogger local_deferredLogger;

        for (std::size_t tracerIdx = 0; tracerIdx < this->tracerPhaseIdx_.size(); ++tracerIdx) {
            if (this->tracerPhaseIdx_[tracerIdx] == FluidSystem::waterPhaseIdx) {
                if (! FluidSystem::phaseIsActive(FluidSystem::waterPhaseIdx)){
                    throw std::runtime_error("Water tracer specified for non-water fluid system: " +
                                             this->name(tracerIdx));
                }

                wat_.addTracer(tracerIdx, this->tracerConcentration_[tracerIdx]);
            }
            else if (this->tracerPhaseIdx_[tracerIdx] == FluidSystem::oilPhaseIdx) {
                if (! FluidSystem::phaseIsActive(FluidSystem::oilPhaseIdx)){
                    throw std::runtime_error("Oil tracer specified for non-oil fluid system: " +
                                             this->name(tracerIdx));
                }

                oil_.addTracer(tracerIdx, this->tracerConcentration_[tracerIdx]);
            }
            else if (this->tracerPhaseIdx_[tracerIdx] == FluidSystem::gasPhaseIdx) {
                if (! FluidSystem::phaseIsActive(FluidSystem::gasPhaseIdx)){
                    throw std::runtime_error("Gas tracer specified for non-gas fluid system: " +
                                             this->name(tracerIdx));
                }

                gas_.addTracer(tracerIdx, this->tracerConcentration_[tracerIdx]);
            }

            // resize free and solution volume storages
            vol1_[0][this->tracerPhaseIdx_[tracerIdx]].
                resize(this->freeTracerConcentration_[tracerIdx].size());
            vol1_[1][this->tracerPhaseIdx_[tracerIdx]].
                resize(this->freeTracerConcentration_[tracerIdx].size());
            dVol_[0][this->tracerPhaseIdx_[tracerIdx]].
                resize(this->solTracerConcentration_[tracerIdx].size());
            dVol_[1][this->tracerPhaseIdx_[tracerIdx]].
                resize(this->solTracerConcentration_[tracerIdx].size());
        }

        if (this->numTracers() != 0) {
            initLinearSystem_();
        }

        this->buildAquiferTracerConnections_(local_deferredLogger);

        const auto& comm = simulator_.vanguard().grid().comm();
        auto global_logger = gatherDeferredLogger(local_deferredLogger, comm);
        if (comm.rank() == 0) {
            global_logger.logMessages();
        }
    }

    void beginTimeStep()
    {
        if (this->numTracers() == 0) {
            return;
        }

        OPM_TIMEBLOCK(tracerUpdateCache);
        updateStorageCache();
    }

    /*!
     * \brief Informs the tracer model that a time step has just been finished.
     */
    void endTimeStep()
    {
        if (this->numTracers() == 0) {
            return;
        }

        OPM_TIMEBLOCK(tracerAdvance);
        advanceTracerFields();
    }

    /*!
     * \brief This method writes the complete state of all tracer
     *        to the hard disk.
     */
    template <class Restarter>
    void serialize(Restarter&)
    { /* not implemented */ }

    /*!
     * \brief This method restores the complete state of the tracer
     *        from disk.
     *
     * It is the inverse of the serialize() method.
     */
    template <class Restarter>
    void deserialize(Restarter&)
    { /* not implemented */ }

    template<class Serializer>
    void serializeOp(Serializer& serializer)
    {
        serializer(static_cast<BaseType&>(*this));
        serializer(tbatch);
    }

protected:
    using TracerTypeIdx = typename BaseType::TracerTypeIdx;
    using BaseType::Free;
    using BaseType::Solution;

    // compute volume associated with free/solution concentration
    template<TracerTypeIdx Index>
    Scalar computeVolume_(const int tracerPhaseIdx,
                          const unsigned globalDofIdx,
                          const unsigned timeIdx) const
    {
        const auto& intQuants = simulator_.model().intensiveQuantities(globalDofIdx, timeIdx);
        const auto& fs = intQuants.fluidState();
        constexpr Scalar min_volume = 1e-10;

        if constexpr (Index == Free) {
            return std::max(decay<Scalar>(fs.saturation(tracerPhaseIdx)) *
                            decay<Scalar>(fs.invB(tracerPhaseIdx)) *
                            decay<Scalar>(intQuants.porosity()),
                            min_volume);
        } else {
            // vaporized oil
            if (tracerPhaseIdx == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) {
                return std::max(decay<Scalar>(fs.saturation(FluidSystem::gasPhaseIdx)) *
                                decay<Scalar>(fs.invB(FluidSystem::gasPhaseIdx)) *
                                decay<Scalar>(fs.Rv()) *
                                decay<Scalar>(intQuants.porosity()),
                                min_volume);
            }

            // dissolved gas
            else if (tracerPhaseIdx == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas()) {
                return std::max(decay<Scalar>(fs.saturation(FluidSystem::oilPhaseIdx)) *
                                decay<Scalar>(fs.invB(FluidSystem::oilPhaseIdx)) *
                                decay<Scalar>(fs.Rs()) *
                                decay<Scalar>(intQuants.porosity()),
                                min_volume);
            }

            return min_volume;
        }
    }

    /*!
     * \brief Set up the neighbor table and the matrix of the tracer equations.
     *
     * Only static data is stored for the faces. The transmissibilities and threshold
     * pressures are taken from the problem during the assembly since they may change
     * during the simulation.
     */
    void initLinearSystem_()
    {
        const auto& problem = simulator_.problem();
        const auto& model = simulator_.model();
        const Scalar gravity = problem.gravity()[GridView::dimensionworld - 1];
        system_.template init<Stencil>(this->gridView_, this->dofMapper_, model.numGridDof(),
            [&problem, &model, gravity](const unsigned globI, const unsigned globJ, const auto& scvf)
            {
                const auto dirId = scvf.dirId();
                return FaceInfo{(problem.dofCenterDepth(globI) - problem.dofCenterDepth(globJ)) * gravity,
                                model.dofTotalVolume(globI),
                                model.dofTotalVolume(globJ),
                                dirId < 0 ? FaceDir::DirEnum::Unknown
                                          : FaceDir::FromIntersectionIndex(dirId)};
            });
    }

    /*!
     * \brief Reservoir volume rate of a phase from cell I to its neighbor across a face.
     *
     * This is the two-point flux of the flow equations (see NewTranFluxModule),
     * multiplied by the face area.
     *
     * \return The rate and whether cell I is upstream.
     */
    std::pair<Scalar, bool>
    phaseVolumeFlux_(const unsigned phaseIdx,
                     const unsigned I,
                     const NeighborInfo& nbInfo) const
    {
        const auto& problem = simulator_.problem();
        const unsigned J = nbInfo.neighbor;
        const auto& face = nbInfo.res_nbinfo;
        const IntensiveQuantities& intQuantsIn = simulator_.model().intensiveQuantities(I, /*timeIdx=*/0);
        const IntensiveQuantities& intQuantsEx = simulator_.model().intensiveQuantities(J, /*timeIdx=*/0);

        constexpr short interiorDofIdx = 0;
        constexpr short exteriorDofIdx = 1;
        short upIdx = interiorDofIdx;
        short dnIdx = exteriorDofIdx;
        Evaluation pressureDifference;
        ExtensiveQuantities::calculatePhasePressureDiff_(upIdx,
                                                         dnIdx,
                                                         pressureDifference,
                                                         intQuantsIn,
                                                         intQuantsEx,
                                                         phaseIdx,
                                                         interiorDofIdx,
                                                         exteriorDofIdx,
                                                         face.Vin,
                                                         face.Vex,
                                                         I,
                                                         J,
                                                         face.dZg,
                                                         problem.thresholdPressure(I, J),
                                                         problem.thresholdPressure(J, I),
                                                         problem.moduleParams());

        const bool inIsUp = upIdx == interiorDofIdx;
        const IntensiveQuantities& up = inIsUp ? intQuantsIn : intQuantsEx;
        // Arithmetic average, as in the flow equations
        const Scalar transMult = (decay<Scalar>(intQuantsIn.rockCompTransMultiplier()) +
                                  decay<Scalar>(intQuantsEx.rockCompTransMultiplier())) / 2;
        const Scalar trans = problem.transmissibility(I, J);
        const Scalar rate = -decay<Scalar>(pressureDifference) *
                            decay<Scalar>(up.mobility(phaseIdx, face.faceDir)) *
                            transMult * trans;
        return {rate, inIsUp};
    }

    //! \brief Surface volume rate of free/solution tracer phase out of cell I across a face.
    template<TracerTypeIdx Index>
    FaceFlux computeFlux_(const int tracerPhaseIdx,
                          const unsigned I,
                          const NeighborInfo& nbInfo) const
    {
        // The flowing phase carrying the tracer
        int flowPhaseIdx = tracerPhaseIdx;
        if constexpr (Index == Solution) {
            if (tracerPhaseIdx == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) {
                // vaporized oil
                flowPhaseIdx = FluidSystem::gasPhaseIdx;
            }
            else if (tracerPhaseIdx == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas()) {
                // dissolved gas
                flowPhaseIdx = FluidSystem::oilPhaseIdx;
            }
            else {
                return {};
            }
        }

        const auto [volumeRate, inIsUp] = phaseVolumeFlux_(flowPhaseIdx, I, nbInfo);
        const unsigned upIdx = inIsUp ? I : nbInfo.neighbor;
        const auto& fs = simulator_.model().intensiveQuantities(upIdx, /*timeIdx=*/0).fluidState();
        Scalar rate = volumeRate * decay<Scalar>(fs.invB(flowPhaseIdx));
        if constexpr (Index == Solution) {
            rate *= flowPhaseIdx == FluidSystem::gasPhaseIdx ? decay<Scalar>(fs.Rv())
                                                             : decay<Scalar>(fs.Rs());
        }
        return {rate, inIsUp};
    }

    template<TracerTypeIdx Index, class TrRe>
    Scalar storage1_(const TrRe& tr,
                     const unsigned tIdx,
                     const unsigned I,
                     const unsigned I1,
                     const bool cache)
    {
        if (cache) {
            return tr.storageOfTimeIndex1_[tIdx][I][Index];
        } else {
            return computeVolume_<Index>(tr.phaseIdx_, I1, 1) *
                   tr.concentration_[tIdx][I1][Index];
        }
    }

    template<class TrRe>
    void assembleTracerEquationVolume(TrRe& tr,
                                      const Scalar scvVolume,
                                      const Scalar dt,
                                      const unsigned I,
                                      const bool enableStorageCache)
    {
        const Scalar fVol = computeVolume_<Free>(tr.phaseIdx_, I, 0);
        const Scalar sVol = computeVolume_<Solution>(tr.phaseIdx_, I, 0);
        dVol_[Solution][tr.phaseIdx_][I] += sVol * scvVolume - vol1_[1][tr.phaseIdx_][I];
        dVol_[Free][tr.phaseIdx_][I] += fVol * scvVolume - vol1_[0][tr.phaseIdx_][I];
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            // Free part
            const Scalar fStorageOfTimeIndex0 = fVol * tr.concentration_[tIdx][I][Free];
            const Scalar fLocalStorage = (fStorageOfTimeIndex0 - storage1_<Free>(tr, tIdx, I, I,
                                                                                 enableStorageCache)) * scvVolume / dt;
            tr.residual_[tIdx][I][Free] += fLocalStorage; // residual + flux

            // Solution part
            const Scalar sStorageOfTimeIndex0 = sVol * tr.concentration_[tIdx][I][Solution];
            const Scalar sLocalStorage = (sStorageOfTimeIndex0 - storage1_<Solution>(tr, tIdx, I, I,
                                                                                     enableStorageCache)) * scvVolume / dt;
            tr.residual_[tIdx][I][Solution] += sLocalStorage; // residual + flux
        }

        // Derivative matrix
        auto& diag = system_.diagonal(I);
        diag[Free][Free] += fVol * scvVolume / dt;
        diag[Solution][Solution] += sVol * scvVolume / dt;
    }

    /*!
     * \brief Add the advective flux of a tracer phase out of cell I across a face.
     *
     * The tracer leaves with the concentration of cell I if cell I is upstream. Otherwise
     * it enters with the concentration given by \c upstreamConcentration(tIdx), which is
     * the concentration of the neighbor for interior faces. Only the dependency on the
     * concentration of cell I is added to the matrix.
     *
     * This is meant to be used for boundary faces as well, with the concentration of the
     * inflowing fluid given by the boundary condition.
     */
    template<TracerTypeIdx Index, class TrRe, class UpstreamConcentration>
    void addFaceFlux_(TrRe& tr,
                      const unsigned I,
                      const FaceFlux& flux,
                      const Scalar dt,
                      UpstreamConcentration&& upstreamConcentration)
    {
        dVol_[Index][tr.phaseIdx_][I] += flux.rate * dt;
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            const Scalar concentration = flux.upIsInterior ? tr.concentration_[tIdx][I][Index]
                                                           : upstreamConcentration(tIdx);
            tr.residual_[tIdx][I][Index] += flux.rate * concentration; // residual + flux
        }
        if (flux.upIsInterior) {
            system_.diagonal(I)[Index][Index] += flux.rate;
        }
    }

    template<class TrRe>
    void assembleTracerEquationFlux(TrRe& tr,
                                    const unsigned I,
                                    const NeighborInfo& nbInfo,
                                    const Scalar dt)
    {
        const unsigned J = nbInfo.neighbor;
        const FaceFlux fFlux = computeFlux_<Free>(tr.phaseIdx_, I, nbInfo);
        const FaceFlux sFlux = computeFlux_<Solution>(tr.phaseIdx_, I, nbInfo);
        addFaceFlux_<Free>(tr, I, fFlux, dt,
                           [&tr, J](const int tIdx) { return tr.concentration_[tIdx][J][Free]; });
        addFaceFlux_<Solution>(tr, I, sFlux, dt,
                               [&tr, J](const int tIdx) { return tr.concentration_[tIdx][J][Solution]; });

        // Derivative of the residual of the neighbor with respect to the concentration of
        // cell I. Each block (J, I) is only written when assembling cell I.
        auto& offDiag = *nbInfo.matBlockAddress;
        if (fFlux.upIsInterior) {
            offDiag[Free][Free] = -fFlux.rate;
        }
        if (sFlux.upIsInterior) {
            offDiag[Solution][Solution] = -sFlux.rate;
        }
    }

    template<class TrRe, class Well>
    void assembleTracerEquationWell(TrRe& tr,
                                    const Well& well)
    {
        if (tr.numTracer() == 0) {
            return;
        }

        const auto& eclWell = well.wellEcl();

        // Init. well output to zero
        auto& tracerRate = this->wellTracerRate_[eclWell.seqIndex()];
        auto& solTracerRate = this->wellSolTracerRate_[eclWell.seqIndex()];
        auto& freeTracerRate = this->wellFreeTracerRate_[eclWell.seqIndex()];
        auto* mswTracerRate = eclWell.isMultiSegment()
            ? &this->mSwTracerRate_[eclWell.seqIndex()]
            : nullptr;
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            tracerRate[tr.idx_[tIdx]] = {this->name(tr.idx_[tIdx]), 0.0};
            freeTracerRate[tr.idx_[tIdx]] = {this->wellfname(tr.idx_[tIdx]), 0.0};
            solTracerRate[tr.idx_[tIdx]] = {this->wellsname(tr.idx_[tIdx]), 0.0};
            if (eclWell.isMultiSegment()) {
                auto& wtr = mswTracerRate->at(tr.idx_[tIdx]) = {this->name(tr.idx_[tIdx])};;
                wtr.rate.reserve(eclWell.getConnections().size());
                for (std::size_t i = 0; i < eclWell.getConnections().size(); ++i) {
                    wtr.rate.emplace(eclWell.getConnections().get(i).segment(), 0.0);
                }
            }
        }

        std::vector<Scalar> wtracer(tr.numTracer());
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            wtracer[tIdx] = this->currentConcentration_(eclWell, this->name(tr.idx_[tIdx]),
                                                        simulator_.problem().wellModel().summaryState());
        }

        const Scalar dt = simulator_.timeStepSize();
        const auto& ws = simulator_.problem().wellModel().wellState().well(well.name());
        const auto well_eff = well.wellEfficiencyFactor(); // Needed to convert ws.phase_mixing_rates to effective rates
        for (std::size_t i = 0; i < ws.perf_data.size(); ++i) {
            const auto I = ws.perf_data.cell_index[i];
            const Scalar rate = well.volumetricSurfaceRateForConnection(I, tr.phaseIdx_); // Includes (accumulated) well efficiency factor
            Scalar rate_s;

            if (tr.phaseIdx_ == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) {
                rate_s = ws.perf_data.phase_mixing_rates[i][ws.vaporized_oil] * well_eff;
            }
            else if (tr.phaseIdx_ == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas()) {
                rate_s = ws.perf_data.phase_mixing_rates[i][ws.dissolved_gas] * well_eff;
            }
            else {
                rate_s = 0.0;
            }

            const Scalar rate_f = rate - rate_s;
            if (rate_f > 0) {
                for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                    const Scalar delta = rate_f * wtracer[tIdx];
                    // Injection of free tracer only
                    tr.residual_[tIdx][I][Free] -= delta;

                    // Store _injector_ tracer rate for reporting
                    // (can be done here since WTRACER is constant)
                    tracerRate[tr.idx_[tIdx]].rate += delta;
                    freeTracerRate[tr.idx_[tIdx]].rate += delta;
                    if (eclWell.isMultiSegment()) {
                        (*mswTracerRate)[tr.idx_[tIdx]].rate[eclWell.getConnections().get(i).segment()] += delta;
                    }
                }
                dVol_[Free][tr.phaseIdx_][I] -= rate_f * dt;
            }
            else if (rate_f < 0) {
                for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                    const Scalar delta = rate_f * wtracer[tIdx];
                    // Store _injector_ tracer rate for cross-flowing well connections
                    // (can be done here since WTRACER is constant)
                    tracerRate[tr.idx_[tIdx]].rate += delta;
                    freeTracerRate[tr.idx_[tIdx]].rate += delta;

                    // Production of free tracer
                    tr.residual_[tIdx][I][Free] -= rate_f * tr.concentration_[tIdx][I][Free];
                }
                dVol_[Free][tr.phaseIdx_][I] -= rate_f * dt;

                // Derivative matrix for free tracer producer
                system_.diagonal(I)[Free][Free] -= rate_f;
            }
            if (rate_s < 0) {
                for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                    // Production of solution tracer
                    tr.residual_[tIdx][I][Solution] -= rate_s * tr.concentration_[tIdx][I][Solution];
                }
                dVol_[Solution][tr.phaseIdx_][I] -= rate_s * dt;

                // Derivative matrix for solution tracer producer
                system_.diagonal(I)[Solution][Solution] -= rate_s;
            }
        }
    }

    template<class TrRe>
    void assembleTracerEquationSource(TrRe& tr,
                                      const Scalar dt,
                                      unsigned I)
    {
        if (tr.numTracer() == 0) {
            return;
        }

        // Skip if solution tracers do not exist
        if (tr.phaseIdx_ ==  FluidSystem::waterPhaseIdx ||
            (tr.phaseIdx_ ==  FluidSystem::gasPhaseIdx && !FluidSystem::enableDissolvedGas()) ||
            (tr.phaseIdx_ ==  FluidSystem::oilPhaseIdx && !FluidSystem::enableVaporizedOil()))
        {
            return;
        }

        const Scalar& dsVol = dVol_[Solution][tr.phaseIdx_][I];
        const Scalar& dfVol = dVol_[Free][tr.phaseIdx_][I];

        // Source term determined by sign of dsVol: if dsVol > 0 then ms -> mf, else mf -> ms
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            if (dsVol >= 0) {
                const auto delta = (dfVol / dt) * tr.concentration_[tIdx][I][Free];
                tr.residual_[tIdx][I][Free] -= delta;
                tr.residual_[tIdx][I][Solution] += delta;
            }
            else {
                const auto delta = (dsVol / dt) * tr.concentration_[tIdx][I][Solution];
                tr.residual_[tIdx][I][Free] += delta;
                tr.residual_[tIdx][I][Solution] -= delta;
            }
        }

        // Derivative matrix
        auto& diag = system_.diagonal(I);
        if (dsVol >= 0) {
            const auto delta = dfVol / dt;
            diag[Free][Free] -= delta;
            diag[Solution][Free] += delta;
        }
        else {
            const auto delta = dsVol / dt;
            diag[Free][Solution] += delta;
            diag[Solution][Solution] -= delta;
        }
    }

    //! \brief Reset the well tracer rates before assembling the tracer equations.
    void prepareWellTracerRates_()
    {
        this->wellTracerRate_.clear();
        this->wellFreeTracerRate_.clear();
        this->wellSolTracerRate_.clear();

        // educated guess for new container size
        const auto num_msw = this->mSwTracerRate_.size();
        this->mSwTracerRate_.clear();

        const auto& wellPtrs = simulator_.problem().wellModel().localNonshutWells();
        this->wellTracerRate_.reserve(wellPtrs.size());
        this->wellFreeTracerRate_.reserve(wellPtrs.size());
        this->wellSolTracerRate_.reserve(wellPtrs.size());
        this->mSwTracerRate_.reserve(num_msw);
        for (const auto& wellPtr : wellPtrs) {
            // Resize vectors of well tracer rates to total number of tracers
            const auto& eclWell = wellPtr->wellEcl();
            this->wellTracerRate_[eclWell.seqIndex()].resize(this->numTracers());
            this->wellFreeTracerRate_[eclWell.seqIndex()].resize(this->numTracers());
            this->wellSolTracerRate_[eclWell.seqIndex()].resize(this->numTracers());
            if (eclWell.isMultiSegment()) {
                this->mSwTracerRate_[eclWell.seqIndex()].resize(this->numTracers());
            }
        }
    }

    /*!
     * \brief Assemble the equations of the tracers of a batch.
     *
     * The equations of the batches are independent, so they share the matrix and are
     * assembled and solved one batch at a time.
     */
    template<class TrRe>
    void assembleTracerEquations_(TrRe& tr)
    {
        // Note that we formulate the equations in terms of a concentration update
        // (compared to previous time step) and not absolute concentration.
        // This implies that current concentration (tr.concentration_[][]) contributes
        // to the rhs both through storage and flux terms.
        // Compare also advanceTracerFields(...) below.

        DeferredLogger local_deferredLogger{};
        OPM_BEGIN_PARALLEL_TRY_CATCH()
        {
            OPM_TIMEBLOCK(tracerAssemble);
            system_.clearMatrix();
            for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                tr.residual_[tIdx] = 0.0;
            }

            // Well terms
            for (const auto& wellPtr : simulator_.problem().wellModel().localNonshutWells()) {
                this->assembleTracerEquationWell(tr, *wellPtr);
            }

            this->assembleTracerEquationAquifer_(tr);

            const Scalar dt = simulator_.timeStepSize();
            const bool enableStorageCache = simulator_.model().enableStorageCache();
            const unsigned numCells = system_.numCells();

            // This is safe as each cell I only writes to its own row of the residual and
            // the matrix, and to the blocks (J, I) of its neighbors J.
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (unsigned I = 0; I < numCells; ++I) {
                if (!system_.isInterior(I)) {
                    // Dirichlet boundary conditions for parallel matrix
                    auto& diag = system_.diagonal(I);
                    diag[Free][Free] = 1.;
                    diag[Solution][Solution] = 1.;
                    continue;
                }

                const Scalar scvVolume = simulator_.model().dofTotalVolume(I);
                this->assembleTracerEquationVolume(tr, scvVolume, dt, I, enableStorageCache);

                for (const auto& nbInfo : system_.neighbors(I)) {
                    this->assembleTracerEquationFlux(tr, I, nbInfo, dt);
                }

                // Boundary conditions (BCCON/BCPROP) are not applied to the tracers, i.e.
                // tracers do not cross the boundary of the domain. Boundary fluxes would be
                // added here with addFaceFlux_(), before the source terms which use the
                // total change of the free and solution volumes of the cell.

                // Source terms (mass transfer between free and solution tracer)
                this->assembleTracerEquationSource(tr, dt, I);
            }
        }
        OPM_END_PARALLEL_TRY_CATCH_LOG(local_deferredLogger,
                                       "assembleTracerEquations() failed: ",
                                       true, simulator_.gridView().comm())

        // Communicate overlap using grid Communication
        auto handle = VectorVectorDataHandle<GridView, std::vector<TracerVector>>(tr.residual_,
                                                                                  simulator_.gridView());
        simulator_.gridView().communicate(handle, Dune::InteriorBorder_All_Interface,
                                          Dune::ForwardCommunication);
    }

    template<TracerTypeIdx Index, class TrRe>
    void updateElem(TrRe& tr,
                    const Scalar scvVolume,
                    const unsigned globalDofIdx)
    {
        const Scalar vol1 = computeVolume_<Index>(tr.phaseIdx_, globalDofIdx, 0);
        vol1_[Index][tr.phaseIdx_][globalDofIdx] = vol1 * scvVolume;
        dVol_[Index][tr.phaseIdx_][globalDofIdx] = 0.0;
        for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
            tr.storageOfTimeIndex1_[tIdx][globalDofIdx][Index] =
                vol1 * tr.concentrationInitial_[tIdx][globalDofIdx][Index];
        }
    }

    void updateStorageCache()
    {
        for (auto& tr : tbatch) {
            if (tr.numTracer() != 0) {
                tr.concentrationInitial_ = tr.concentration_;
            }
        }

        const unsigned numCells = simulator_.model().numGridDof();

        // This is safe as each cell only writes to its own entries in the shared arrays.
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (unsigned globalDofIdx = 0; globalDofIdx < numCells; ++globalDofIdx) {
            const Scalar scvVolume = simulator_.model().dofTotalVolume(globalDofIdx);
            for (auto& tr : tbatch) {
                if (tr.numTracer() == 0) {
                    continue;
                }
                updateElem<Free>(tr, scvVolume, globalDofIdx);
                updateElem<Solution>(tr, scvVolume, globalDofIdx);
            }
        }
    }

    template<TracerTypeIdx Index, class TrRe>
    void copyForOutput(TrRe& tr,
                       const std::vector<TracerVector>& dx,
                       const Scalar S,
                       const unsigned tIdx,
                       const unsigned globalDofIdx,
                       std::vector<TracerVectorSingle>& sc)
    {
        constexpr Scalar tol_gas_sat = 1e-6;
        tr.concentration_[tIdx][globalDofIdx][Index] -= dx[tIdx][globalDofIdx][Index];
        if (tr.concentration_[tIdx][globalDofIdx][Index] < 0.0 || S < tol_gas_sat) {
            tr.concentration_[tIdx][globalDofIdx][Index] = 0.0;
        }
        sc[tr.idx_[tIdx]][globalDofIdx] = tr.concentration_[tIdx][globalDofIdx][Index];
    }

    template<TracerTypeIdx Index, class TrRe>
    void assignRates(const TrRe& tr,
                     const Well& eclWell,
                     const std::size_t i,
                     const std::size_t I,
                     const Scalar rate,
                     std::vector<WellTracerRate<Scalar>>& tracerRate,
                     std::vector<MSWellTracerRate<Scalar>>* mswTracerRate,
                     std::vector<WellTracerRate<Scalar>>& splitRate)
    {
        if (rate < 0) {
            for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                // Store _producer_ free tracer rate for reporting
                const Scalar delta = rate * tr.concentration_[tIdx][I][Index];
                tracerRate[tr.idx_[tIdx]].rate += delta;
                splitRate[tr.idx_[tIdx]].rate += delta;
                if (eclWell.isMultiSegment()) {
                    (*mswTracerRate)[tr.idx_[tIdx]].rate[eclWell.getConnections().get(i).segment()] += delta;
                }
            }
        }
    }

    void convertEffectiveRatesToRawRates() {
        const auto& wellPtrs = simulator_.problem().wellModel().localNonshutWells();
        for (const auto& wellPtr : wellPtrs) {
            const auto& eclWell = wellPtr->wellEcl();
            const auto well_seq_index = eclWell.seqIndex();
            const auto inv_well_eff_factor = 1.0 / std::max(Scalar{1.0e-10}, wellPtr->wellEfficiencyFactor());

            std::ranges::for_each(this->wellTracerRate_[well_seq_index], [&](WellTracerRate<Scalar>& wtr) {
                wtr.rate *= inv_well_eff_factor;
            });
            std::ranges::for_each(this->wellFreeTracerRate_[well_seq_index], [&](WellTracerRate<Scalar>& wtr) {
                wtr.rate *= inv_well_eff_factor;
            });
            std::ranges::for_each(this->wellSolTracerRate_[well_seq_index], [&](WellTracerRate<Scalar>& wtr) {
                wtr.rate *= inv_well_eff_factor;
            });
            if (eclWell.isMultiSegment()) {
                std::ranges::for_each(this->mSwTracerRate_[well_seq_index], [&](MSWellTracerRate<Scalar>& wtr) {
                    std::ranges::for_each(wtr.rate, [&](auto& item) {
                        item.second *= inv_well_eff_factor;
                    });
                });
            }
        }
    }

    void buildAquiferTracerConnections_(DeferredLogger& deferredLogger)
    {
        aquifer_tracer_cells_.clear();

        const auto& aquifer_cfg = this->eclState_.aquifer();
        if (!aquifer_cfg.active()) {
            return;
        }

        const auto& specs = aquifer_cfg.aquiferTracers();
        if (specs.empty()) {
            return;
        }

        std::unordered_map<std::string, int> tracer_name_to_idx;
        for (int tracerIdx = 0; tracerIdx < this->numTracers(); ++tracerIdx) {
            tracer_name_to_idx.emplace(this->name(tracerIdx), tracerIdx);
        }

        const auto& vanguard = simulator_.vanguard();
        for (const auto& spec : specs) {
            if (!aquifer_cfg.hasAnalyticalAquifer(spec.aquifer_id)) {
                continue;
            }

            const auto tracer_pos = tracer_name_to_idx.find(spec.tracer_name);
            if (tracer_pos == tracer_name_to_idx.end()) {
                if (simulator_.vanguard().grid().comm().rank() == 0) {
                    deferredLogger.warning(fmt::format("AQANTRC tracer '{}' is not declared in TRACER",
                                                       spec.tracer_name));
                }
                continue;
            }

            if (!aquifer_cfg.connections().hasAquiferConnections(spec.aquifer_id)) {
                continue;
            }

            const int tracerIdx = tracer_pos->second;
            const int phaseIdx = this->tracerPhaseIdx_[tracerIdx];

            for (const auto& conn : aquifer_cfg.connections().getConnections(spec.aquifer_id)) {
                const int cellIdx = vanguard.compressedIndex(conn.global_index);
                if (cellIdx < 0) {
                    continue;
                }

                aquifer_tracer_cells_[cellIdx].push_back(
                    AquiferTracerCellSpec { tracerIdx, phaseIdx, static_cast<Scalar>(spec.concentration) });
            }
        }
    }

    template<class TrRe>
    void assembleTracerEquationAquifer_(TrRe& tr)
    {
        if (tr.numTracer() == 0 || aquifer_tracer_cells_.empty()) {
            return;
        }

        const auto& aquiferModel = simulator_.problem().aquiferModel();
        const Scalar dt = simulator_.timeStepSize();

        for (const auto& [cellIdx, specs] : aquifer_tracer_cells_) {
            // Scalar influx from converged Qai_; do not recalculate or use getValue()
            // in the flow Newton path (see AquiferAnalytical::addToSource).
            const Scalar rate = aquiferModel.cachedConnectionInfluxRate(cellIdx);
            if (rate == Scalar{0}) {
                continue;
            }

            const unsigned I = cellIdx;
            const Scalar rate_f = rate;

            for (const auto& spec : specs) {
                if (spec.phaseIdx != tr.phaseIdx_) {
                    continue;
                }

                int localTIdx = -1;
                for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                    if (tr.idx_[tIdx] == spec.tracerIdx) {
                        localTIdx = tIdx;
                        break;
                    }
                }
                if (localTIdx < 0) {
                    continue;
                }

                if (rate_f > 0) {
                    const Scalar delta = rate_f * spec.concentration;
                    tr.residual_[localTIdx][I][Free] -= delta;
                    dVol_[Free][tr.phaseIdx_][I] -= rate_f * dt;
                }
                else if (rate_f < 0) {
                    tr.residual_[localTIdx][I][Free] -= rate_f * tr.concentration_[localTIdx][I][Free];
                    dVol_[Free][tr.phaseIdx_][I] -= rate_f * dt;
                    system_.diagonal(I)[Free][Free] -= rate_f;
                }
            }
        }
    }

    void advanceTracerFields()
    {
        prepareWellTracerRates_();

        for (auto& tr : tbatch) {
            if (tr.numTracer() == 0) {
                continue;
            }

            assembleTracerEquations_(tr);

            // Note that we solve for a concentration update (compared to previous time step)
            // Confer also assembleTracerEquations_(...) above.
            std::vector<TracerVector> dx(tr.concentration_);
            for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                dx[tIdx] = 0.0;
            }

            const bool converged = this->linearSolveBatchwise_(system_.matrix(), dx, tr.residual_);
            if (!converged) {
                OpmLog::warning("### Tracer model: Linear solver did not converge. ###");
            }

            OPM_TIMEBLOCK(tracerPost);

            for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                for (std::size_t globalDofIdx = 0; globalDofIdx < tr.concentration_[tIdx].size(); ++globalDofIdx) {
                    // New concetration. Concentrations that are negative or where free/solution phase is not
                    // present are set to zero
                    const auto& intQuants = simulator_.model().intensiveQuantities(globalDofIdx, 0);
                    const auto& fs = intQuants.fluidState();
                    const Scalar Sf = decay<Scalar>(fs.saturation(tr.phaseIdx_));
                    Scalar Ss = 0.0;

                    if (tr.phaseIdx_ == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas()) {
                        Ss = decay<Scalar>(fs.saturation(FluidSystem::oilPhaseIdx));
                    }
                    else if (tr.phaseIdx_ == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) {
                        Ss = decay<Scalar>(fs.saturation(FluidSystem::gasPhaseIdx));
                    }

                    copyForOutput<Free>(tr, dx, Sf, tIdx, globalDofIdx, this->freeTracerConcentration_);
                    copyForOutput<Solution>(tr, dx, Ss, tIdx, globalDofIdx, this->solTracerConcentration_);
                }
            }

            // Store _producer_ tracer rate for reporting
            const auto& wellPtrs = simulator_.problem().wellModel().localNonshutWells();
            for (const auto& wellPtr : wellPtrs) {
                const auto& eclWell = wellPtr->wellEcl();

                // Injection rates already reported during assembly
                if (!eclWell.isProducer()) {
                    continue;
                }

                Scalar rateWellPos = 0.0;
                Scalar rateWellNeg = 0.0;
                const std::size_t well_index = simulator_.problem().wellModel().wellState().index(eclWell.name()).value();
                const auto& ws = simulator_.problem().wellModel().wellState().well(well_index);
                auto& tracerRate = this->wellTracerRate_[eclWell.seqIndex()];
                auto& freeTracerRate = this->wellFreeTracerRate_[eclWell.seqIndex()];
                auto& solTracerRate = this->wellSolTracerRate_[eclWell.seqIndex()];
                auto* mswTracerRate = eclWell.isMultiSegment() ? &this->mSwTracerRate_[eclWell.seqIndex()] : nullptr;
                const auto well_eff = wellPtr->wellEfficiencyFactor(); // Needed to convert ws.phase_mixing_rates to effective rates
                for (std::size_t i = 0; i < ws.perf_data.size(); ++i) {
                    const auto I = ws.perf_data.cell_index[i];
                    const Scalar rate = wellPtr->volumetricSurfaceRateForConnection(I, tr.phaseIdx_); // Includes (accumulated) well efficiency factor

                    Scalar rate_s;
                    if (tr.phaseIdx_ == FluidSystem::oilPhaseIdx && FluidSystem::enableVaporizedOil()) {
                        rate_s = ws.perf_data.phase_mixing_rates[i][ws.vaporized_oil]*well_eff;
                    }
                    else if (tr.phaseIdx_ == FluidSystem::gasPhaseIdx && FluidSystem::enableDissolvedGas()) {
                        rate_s = ws.perf_data.phase_mixing_rates[i][ws.dissolved_gas]*well_eff;
                    }
                    else {
                        rate_s = 0.0;
                    }

                    const Scalar rate_f = rate - rate_s;
                    assignRates<Free>(tr, eclWell, i, I, rate_f,
                                      tracerRate, mswTracerRate, freeTracerRate);
                    assignRates<Solution>(tr, eclWell, i, I, rate_s,
                                          tracerRate, mswTracerRate, solTracerRate);

                    if (rate < 0) {
                        rateWellNeg += rate;
                    }
                    else {
                        rateWellPos += rate;
                    }
                }

                // TODO: Some inconsistencies here that perhaps should be clarified.
                // The "offical" rate as reported below is occasionally significant
                // different from the sum over connections (as calculated above). Only observed
                // for small values, neglible for the rate itself, but matters when used to
                // calculate tracer concentrations.
                const Scalar official_well_rate_total =
                    simulator_.problem().wellModel().wellState().well(well_index).surface_rates[tr.phaseIdx_];

                const Scalar rateWellTotal = official_well_rate_total;

                if (rateWellTotal > rateWellNeg) { // Cross flow
                    constexpr Scalar bucketPrDay = 10.0 / (1000. * 3600. * 24.); // ... keeps (some) trouble away
                    const Scalar factor = (rateWellTotal < -bucketPrDay) ? rateWellTotal / rateWellNeg : 0.0;
                    for (int tIdx = 0; tIdx < tr.numTracer(); ++tIdx) {
                        tracerRate[tIdx].rate *= factor;
                    }
                }
            }
        }
        convertEffectiveRatesToRawRates();
    }

    Simulator& simulator_;

    // This struct collects tracers of the same type (i.e, transported in same phase).
    // The idea being that, under the assumption of linearity, tracers of same type can
    // be solved in concert, having a common system matrix but separate right-hand-sides.

    // Since oil or gas tracers appears in dual compositions when VAPOIL respectively DISGAS
    // is active, the template argument is intended to support future extension to these
    // scenarios by supplying an extended vector type.

    template <typename TV>
    struct TracerBatch
    {
        std::vector<int> idx_;
        const int phaseIdx_;
        std::vector<TV> concentrationInitial_;
        std::vector<TV> concentration_;
        std::vector<TV> storageOfTimeIndex1_;
        std::vector<TV> residual_;

        bool operator==(const TracerBatch& rhs) const
        {
            return this->concentrationInitial_ == rhs.concentrationInitial_ &&
                   this->concentration_ == rhs.concentration_;
        }

        static TracerBatch serializationTestObject()
        {
            TracerBatch<TV> result(4);
            result.idx_ = {1,2,3};
            result.concentrationInitial_ = {5.0, 6.0};
            result.concentration_ = {7.0, 8.0};
            result.storageOfTimeIndex1_ = {9.0, 10.0, 11.0};
            result.residual_ = {12.0, 13.0};

            return result;
        }

        template<class Serializer>
        void serializeOp(Serializer& serializer)
        {
            serializer(concentrationInitial_);
            serializer(concentration_);
        }

        TracerBatch(int phaseIdx = 0) : phaseIdx_(phaseIdx) {}

        int numTracer() const
        { return idx_.size(); }

        void addTracer(const int idx, const TV& concentration)
        {
            const int numGridDof = concentration.size();
            idx_.emplace_back(idx);
            concentrationInitial_.emplace_back(concentration);
            concentration_.emplace_back(concentration);
            residual_.emplace_back(numGridDof);
            storageOfTimeIndex1_.emplace_back(numGridDof);
        }
    };

    std::array<TracerBatch<TracerVector>,numPhases> tbatch;
    TracerBatch<TracerVector>& wat_;
    TracerBatch<TracerVector>& oil_;
    TracerBatch<TracerVector>& gas_;
    std::array<std::array<std::vector<Scalar>,numPhases>,2> vol1_;
    std::array<std::array<std::vector<Scalar>,numPhases>,2> dVol_;
    LinearSystem system_{};

    struct AquiferTracerCellSpec {
        int tracerIdx{};
        int phaseIdx{};
        Scalar concentration{};
    };

    std::unordered_map<unsigned, std::vector<AquiferTracerCellSpec>> aquifer_tracer_cells_;
};

} // namespace Opm

#endif // OPM_TRACER_MODEL_HPP
